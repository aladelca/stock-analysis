# GCP On-Demand Cloud Run Job

This runbook deploys the additive GCP pipeline. It does not replace the local `run-one-shot`
workflow.

The cloud flow has two commands:

```bash
stock-analysis train-gcp-model --config configs/portfolio.gcp.yaml --forecast-engine ml
stock-analysis run-gcp-one-shot --config configs/portfolio.gcp.yaml --forecast-engine ml
```

`train-gcp-model` trains/calibrates the ML forecast model and writes a versioned model bundle to
Cloud Storage. By default it also promotes the same bundle to the production model path used by
inference:

```text
gs://<bucket>/models/runs/<training_run_id>/model.cloudpickle
gs://<bucket>/models/runs/<training_run_id>/metadata.json
gs://<bucket>/models/runs/<training_run_id>/calibration_diagnostics.parquet
gs://<bucket>/models/runs/<training_run_id>/calibration_predictions.parquet
gs://<bucket>/models/production/model.cloudpickle
```

`run-gcp-one-shot` loads the configured model artifact from Cloud Storage and does not retrain. If
no production model exists yet, the command fails and tells you to run `train-gcp-model` first.

It writes medallion artifacts directly to Cloud Storage:

```text
gs://<bucket>/runs/<run_id>/raw/
gs://<bucket>/runs/<run_id>/bronze/
gs://<bucket>/runs/<run_id>/silver/
gs://<bucket>/runs/<run_id>/gold/
```

It also publishes Tableau-ready tables to BigQuery when `gcp.publish_bigquery: true`.
Run-scoped tables are appendable and deduplicated by `run_id`; account history tables are
full refreshed for the account so repeated on-demand executions do not duplicate older history rows.

## Preconditions

Install local tools:

```bash
gcloud --version
docker --version
```

Authenticate:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <project_id>
```

Enable APIs:

```bash
gcloud services enable \
  artifactregistry.googleapis.com \
  bigquery.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com
```

## Create Resources

Set variables:

```bash
export PROJECT_ID="<project_id>"
export REGION="us-central1"
export REPOSITORY="stock-analysis"
export IMAGE="stock-analysis"
export BUCKET="stock-analysis-medallion-prod"
export GOLD_DATASET="stock_analysis_gold"
export SERVICE_ACCOUNT="stock-analysis-runner"
```

Create Artifact Registry:

```bash
gcloud artifacts repositories create "${REPOSITORY}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Stock analysis pipeline images"
```

Create the Cloud Storage medallion bucket:

```bash
gcloud storage buckets create "gs://${BUCKET}" \
  --location="${REGION}" \
  --uniform-bucket-level-access
```

Create the BigQuery dataset:

```bash
bq --location=US mk --dataset "${PROJECT_ID}:${GOLD_DATASET}"
```

Create a service account:

```bash
gcloud iam service-accounts create "${SERVICE_ACCOUNT}" \
  --display-name="Stock Analysis Cloud Run Job"
```

Grant minimum roles:

```bash
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.jobUser"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

```

For tighter production IAM, scope BigQuery permissions to the dataset instead of the whole project.

## Configure The Cloud YAML

Edit `configs/portfolio.gcp.yaml`:

```yaml
gcp:
  enabled: true
  project_id: <project_id>
  region: us-central1
  bucket: <bucket>
  gcs_prefix: runs
  model_registry_prefix: models
  model_artifact_uri: null
  bigquery_location: US
  bigquery_dataset_gold: stock_analysis_gold
  publish_bigquery: true
```

Leave `model_artifact_uri: null` for normal operation. The inference job will read:

```text
gs://<bucket>/models/production/model.cloudpickle
```

Set `model_artifact_uri` only when you want to pin inference to a specific versioned model, for
example:

```yaml
gcp:
  model_artifact_uri: gs://<bucket>/models/runs/20260503T150000Z/model.cloudpickle
```

The GCP path should use Cloud Storage and BigQuery only. Leave Supabase disabled:

```yaml
live_account:
  enabled: false
  account_slug: null
  cashflow_source: scenario

supabase:
  enabled: false
```

Account tracking should move to a BigQuery-backed repository before enabling actual cloud
cashflows/snapshots.

## Build And Push

Configure Docker auth:

```bash
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
```

Build:

```bash
docker build \
  -t "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}:latest" \
  .
```

Push:

```bash
docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}:latest"
```

## Create The Cloud Run Jobs

Create the training job:

```bash
gcloud run jobs create stock-analysis-train-model \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}:latest" \
  --region="${REGION}" \
  --service-account="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --memory=8Gi \
  --cpu=4 \
  --task-timeout=3600 \
  --args="train-gcp-model,--config,configs/portfolio.gcp.yaml,--forecast-engine,ml"
```

Create the inference/recommendation job:

```bash
gcloud run jobs create stock-analysis-one-shot \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}:latest" \
  --region="${REGION}" \
  --service-account="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --memory=8Gi \
  --cpu=4 \
  --task-timeout=3600 \
  --args="run-gcp-one-shot,--config,configs/portfolio.gcp.yaml,--forecast-engine,ml"
```

## Execute On Demand

Train and promote the current model:

```bash
gcloud run jobs execute stock-analysis-train-model \
  --region="${REGION}" \
  --wait
```

Then run recommendations from the promoted model:

```bash
gcloud run jobs execute stock-analysis-one-shot \
  --region="${REGION}" \
  --wait
```

Inspect logs:

```bash
gcloud run jobs executions list \
  --job=stock-analysis-train-model \
  --region="${REGION}"

gcloud run jobs executions list \
  --job=stock-analysis-one-shot \
  --region="${REGION}"
```

List GCS outputs:

```bash
gcloud storage ls "gs://${BUCKET}/runs/"
gcloud storage ls "gs://${BUCKET}/models/production/"
gcloud storage ls "gs://${BUCKET}/models/runs/"
```

Training run gold outputs include `model_metadata.parquet` and `model_metadata.csv` under:

```text
gs://<bucket>/runs/<training_run_id>/gold/
```

Inspect BigQuery tables:

```bash
bq ls "${PROJECT_ID}:${GOLD_DATASET}"
```

Expected core tables include:

```text
portfolio_dashboard_mart
optimizer_input
portfolio_recommendations
portfolio_risk_metrics
sector_exposure
run_metadata
forecast_calibration_diagnostics
forecast_calibration_predictions
recommendation_lines_history
performance_snapshots_history
```

## Tableau

Connect Tableau to Google BigQuery and use:

```text
<project_id>.stock_analysis_gold.portfolio_dashboard_mart
<project_id>.stock_analysis_gold.portfolio_recommendations
<project_id>.stock_analysis_gold.recommendation_lines_history
<project_id>.stock_analysis_gold.performance_snapshots_history
```

Use `run_id`, `as_of_date`, and `run_data_as_of_date` as dashboard filters.

## Notes

- The local pipeline still writes to `data/runs/<run_id>/`.
- The GCP pipeline writes directly to Cloud Storage through `GcsArtifactStore`.
- GCP model artifacts are stored directly in Cloud Storage through `GcsModelRegistry`.
- Inference loads the promoted GCS model artifact and refuses to retrain implicitly.
- Cloud Scheduler is intentionally not configured in this phase.
- Tableau Hyper export is skipped in the cloud command; BigQuery is the cloud serving layer.
