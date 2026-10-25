
import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

DATA = Path("data")
SLUG = "manufacturing-predictive-maintenance"
TITLE = "Manufacturing Predictive Maintenance"


def connect():
    """Return (api, dataset) for the logged-in user's default dataset. Raises RuntimeError if login fails."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    try:
        api.authenticate()
    except (Exception, SystemExit):
        raise RuntimeError("Kaggle login failed. Run `kaggle auth login` once, then try again.")
    return api, f"{api.get_config_value('username')}/{SLUG}"


def source_file(folder) -> Path:
    """Marker written after each pull, kept beside the folder so push doesn't upload it."""
    folder = Path(folder)
    return folder.with_name(f".{folder.name}_kaggle_source.json")


def exists(api, dataset) -> bool:
    try:
        api.dataset_status(dataset)
        return True
    except Exception:
        return False


def pull(api, dataset, folder=DATA, quiet=False):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    # file by file: the whole-dataset zip 404s until Kaggle finishes building it after an upload
    for name in [f.name for f in api.dataset_list_files(dataset).files]:
        api.dataset_download_file(dataset, name, path=str(folder), force=True, quiet=quiet)
        zipped = folder / (name + ".zip")
        if zipped.exists():  # Kaggle sometimes serves single files compressed
            with zipfile.ZipFile(zipped) as z:
                z.extractall(folder)
            zipped.unlink()
    source_file(folder).write_text(json.dumps(
        {"dataset": dataset, "pulled_at": datetime.now().isoformat(timespec="seconds")}, indent=2))


def data_source(folder=DATA):
    """Kaggle pull info if the current data/ came from Kaggle, or None if it was generated locally since."""
    folder = Path(folder)
    marker, sensors = source_file(folder), folder / "sensor_readings.parquet"
    if not marker.exists() or not sensors.exists():
        return None
    # generate_dataset.py rewrites the parquet without touching the marker, so a newer parquet means local data
    if sensors.stat().st_mtime > marker.stat().st_mtime + 1:
        return None
    return json.loads(marker.read_text())


def main():
    ap = argparse.ArgumentParser(description="Push/pull the dataset to/from Kaggle")
    ap.add_argument("command", choices=["push", "pull", "status"])
    ap.add_argument("-m", "--message", default="Update dataset", help="version notes for push")
    ap.add_argument("--dataset", help="owner/slug (default <your-username>/" + SLUG + ")")
    ap.add_argument("--public", action="store_true", help="make the dataset public when first created")
    ap.add_argument("--out", default=str(DATA), help="folder to push from / pull into (default ./data)")
    args = ap.parse_args()

    try:
        api, default = connect()
    except RuntimeError as e:
        sys.exit(str(e))
    dataset = args.dataset or default
    folder = Path(args.out)

    if args.command == "push":
        if not folder.is_dir() or not any(folder.iterdir()):
            sys.exit(f"{folder}/ is empty - run python generate_dataset.py first.")
        meta = folder / "dataset-metadata.json"
        if not meta.exists():
            meta.write_text(json.dumps({"title": TITLE, "id": dataset, "licenses": [{"name": "CC0-1.0"}]}, indent=2))
        if exists(api, dataset):
            print(f"Uploading new version of {dataset} ...")
            api.dataset_create_version(str(folder), version_notes=args.message, convert_to_csv=False, dir_mode="zip")
        else:
            print(f"Creating {'public' if args.public else 'private'} dataset {dataset} ...")
            api.dataset_create_new(str(folder), public=args.public, convert_to_csv=False, dir_mode="zip")
        print(f"https://www.kaggle.com/datasets/{dataset}")

    elif args.command == "pull":
        print(f"Downloading {dataset} into {folder}/ ...")
        pull(api, dataset, folder)
        print("Done.")

    else:
        try:
            print(f"{dataset}: {api.dataset_status(dataset)}")
        except Exception:
            print(f"{dataset}: not found (not pushed yet, or no access). Login is OK.")


if __name__ == "__main__":
    main()
