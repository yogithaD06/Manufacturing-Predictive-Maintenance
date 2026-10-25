# Manufacturing Predictive Maintenance

Generates a 2M+ row synthetic big-data dataset, publishes it to Kaggle, trains a failure-prediction model, and shows it all in an interactive dashboard that loads its data from Kaggle.

## Quick start

Already set up? These are the commands in order (details below):

```powershell
.venv\Scripts\activate
kaggle auth login                              # step 1: once per computer
python generate_dataset.py                     # step 2: create data/
python kaggle_sync.py push -m "first upload"   # step 3: publish to Kaggle
python kaggle_sync.py status                   #         wait until "ready"
python train_model.py                          # step 4: train the model
streamlit run app.py                           # step 5: dashboard (data from Kaggle)
```

## How the data flows

```
generate_dataset.py ──► data/ ──push──► Kaggle dataset ──pull / Refresh──► data/ ──► dashboard
                                                                              └──► train_model.py ──► models/
```

- **Kaggle holds the dataset.** The dashboard downloads it from there.
- **The model is local.** `models/` is not uploaded to Kaggle, so train it on each computer, and again whenever the data changes.

## 1. Setup (one time)

You need Python 3.10 or newer. Open this folder in VS Code, then open a terminal (Ctrl + `):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

### Log in to Kaggle (required)

This demo publishes its dataset to Kaggle and reads it back from there, so you need a free Kaggle account (sign up at https://www.kaggle.com). With the venv active, pick one:

- **Browser login (recommended):** `kaggle auth login`. Your login is saved, so you only do this once.
- **API token:** create one at https://www.kaggle.com/settings/api and save it to `%USERPROFILE%\.kaggle\access_token`.

Check it with `python kaggle_sync.py status`:

| Output | Meaning |
|---|---|
| `<you>/manufacturing-predictive-maintenance: not found ...` | Logged in, nothing uploaded yet. Go to step 2. |
| `<you>/manufacturing-predictive-maintenance: ready` | Logged in and your dataset is on Kaggle. |
| `Kaggle login failed ...` | Not logged in. Run `kaggle auth login`. |

## 2. Generate the dataset (~30 seconds)

```powershell
python generate_dataset.py
```

This creates the `data/` folder with 5 tables (2,073,600 sensor readings, 120 machines).

Options:
- `python generate_dataset.py --machines 30 --days 30` — small, fast version for a quick test
- `python generate_dataset.py --csv` — also write `sensor_readings.csv` (~170 MB) for Kaggle users who prefer CSV

## 3. Publish the dataset to Kaggle

```powershell
python kaggle_sync.py push -m "first upload"
python kaggle_sync.py status        # wait until it says "ready"
```

This uploads `data/` to the Kaggle dataset `<your-username>/manufacturing-predictive-maintenance` and prints its link. The first upload creates a **private** dataset; add `--public` to make it public.

## 4. Train the model (~30 seconds)

```powershell
python train_model.py
```

Predicts "will this machine fail in the next 24 hours?", tested on the last 25% of the timeline. It saves to `models/`.

## 5. Run the dashboard

```powershell
streamlit run app.py
```

Your browser opens at http://localhost:8501. Stop it with Ctrl + C in the terminal.

The dashboard reads its data from Kaggle. Everything is in the **sidebar**:

| You see / do | What happens |
|---|---|
| Start the dashboard with an empty `data/` | It downloads your Kaggle dataset automatically. |
| Click **🔄 Refresh from Kaggle** | Downloads the latest version of the dataset and reloads every tab. |
| Type `owner/dataset-name` in **Kaggle dataset**, then Refresh | Shows someone else's dataset. Leave it blank for your own. |
| Green **Kaggle** box | The data on screen came from Kaggle. It links to the dataset and shows the download time. |
| Yellow **Local data** box | The data was generated on this computer and hasn't come from Kaggle. Push it (step 3), then click Refresh. |

After a Refresh that brings in different data, run `python train_model.py` and reload the page so the **Failure risk** tab scores the new data.

## Common workflows

**Change the data and show it from Kaggle:**

```powershell
python generate_dataset.py                       # add --csv to include sensor_readings.csv (~170 MB)
python kaggle_sync.py push -m "regenerated with new settings"
python kaggle_sync.py status                     # wait until it says "ready"
python train_model.py
streamlit run app.py                             # then click Refresh from Kaggle
```

**Run the demo on a new computer (no generating needed):**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
kaggle auth login
python kaggle_sync.py pull
python train_model.py
streamlit run app.py
```

**Show someone else's dataset:**

```powershell
python kaggle_sync.py pull --dataset owner/dataset-name
python train_model.py
streamlit run app.py
```

Or type `owner/dataset-name` in the dashboard sidebar and click Refresh, then run `python train_model.py` and reload the page.

## kaggle_sync.py commands

| Task | Command |
|---|---|
| First upload (creates a **private** dataset) | `python kaggle_sync.py push -m "first upload"` |
| First upload as a **public** dataset | `python kaggle_sync.py push -m "first upload" --public` |
| Upload a new version after regenerating data | `python kaggle_sync.py push -m "what changed"` |
| Download the latest version into `data/` | `python kaggle_sync.py pull` |
| Download into a different folder | `python kaggle_sync.py pull --out kaggle_data` |
| Check processing status after an upload | `python kaggle_sync.py status` |
| Use a different dataset (yours or someone else's) | add `--dataset owner/dataset-name` to any command |
| Show all options | `python kaggle_sync.py --help` |

### Notes

- The first push writes `data/dataset-metadata.json` (title, dataset id, CC0 license). Edit it to change the title or license, then push again.
- Each push replaces every file in the dataset. Kaggle keeps older versions in the dataset's history.
- You can change a dataset between private and public later in its Settings on kaggle.com.
- `pull` (and the dashboard's Refresh) downloads the files one at a time and overwrites files with the same name in `data/`.
- After each pull, `.data_kaggle_source.json` (next to `data/`) records which dataset was downloaded and when. The sidebar reads it. It is never uploaded.
- Raw Kaggle CLI equivalents, if you need them: `kaggle datasets version -p data -m "notes" --dir-mode zip`, `kaggle datasets status <owner>/<name>`. For downloads, prefer `python kaggle_sync.py pull`: `kaggle datasets download` fetches one zip, which can fail with a 404 for a while after an upload.

## Dashboard walkthrough

1. **Overview tab** — the scale (2M+ readings), the 4 plants, failure modes, and the cost of corrective vs preventive maintenance. Corrective repairs cost far more, which is the business case for prediction.
2. **Machine explorer** — pick a machine. Temperature and vibration rise and oil quality drops before each red failure line. That is the signal a model learns.
3. **Failure risk (ML)** — model accuracy, then a live ranking of which machines are likely to fail in the next 24 hours. Move the alert threshold slider.
4. **Data & schema** — the tables and columns, ready for anyone to reuse from Kaggle.

## Files

| File | Purpose |
|---|---|
| `generate_dataset.py` | Simulates machine degradation, failures, maintenance and production (seed 42, reproducible) |
| `kaggle_sync.py` | Upload (push) and download (pull) `data/` to and from Kaggle |
| `train_model.py` | Feature engineering and gradient-boosting classifier |
| `app.py` | Streamlit dashboard (loads its data from Kaggle) |
| `requirements.txt` | Python dependencies |

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found on Windows | Try `py` instead of `python`. |
| Activation blocked in PowerShell | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again. |
| `kaggle` not found | Activate the venv, or run `pip install -r requirements.txt`. |
| "Kaggle login failed" | Run `kaggle auth login` with the venv active. |
| Dashboard says "Kaggle download failed" | Run `kaggle auth login`, and check that step 3 finished (`python kaggle_sync.py status` says "ready"). For someone else's dataset, check the name and that you have access. |
| Sidebar says "Local data" | You generated data after the last download. Push it (step 3), then click **Refresh from Kaggle**. |
| Failure risk tab says "No trained model yet" | Run `python train_model.py`, then reload the page. |
| Push fails with a "title" or "id" error | Check `data/dataset-metadata.json`. The title must be 6–50 characters and the id must be `your-username/slug`. |
