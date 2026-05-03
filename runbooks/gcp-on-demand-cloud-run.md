# GCP On-Demand Cloud Run Job

This runbook deploys the additive GCP pipeline. It does not replace the local `run-one-shot`
workflow.

The cloud flow has two commands:

```bash
stock-analysis train-gcp-model --config configs/portfolio.gcp.yaml --forecast-engine ml
stock-analysis run-gcp-one-shot --config configs/portfolio.gcp.yaml --forecast-engine ml
```

`train-gcp-model` trains/calibrates the ML forecast model and writes a versioned model bundle to
Cloud Storage. By default it also promotes the same immutable run bundle by writing the production
pointer used by inference:

```text
gs://<bucket>/models/runs/<training_run_id>/model.cloudpickle
gs://<bucket>/models/runs/<training_run_id>/metadata.json
gs://<bucket>/models/runs/<training_run_id>/calibration_diagnostics.parquet
gs://<bucket>/models/runs/<training_run_id>/calibration_predictions.parquet
gs://<bucket>/models/runs/<training_run_id>/manifest.json
gs://<bucket>/models/production/current.json
```

`run-gcp-one-shot` loads the configured model artifact from Cloud Storage and does not retrain. If
no complete production manifest exists yet, the command fails and tells you to run
`train-gcp-model` first. Inference validates model version, target column, forecast horizon,
score scale, feature columns, calibration contract, and trained-through date before scoring.

It writes medallion artifacts directly to Cloud Storage:

```text
gs://<bucket>/runs/<run_id>/raw/
gs://<bucket>/runs/<run_id>/bronze/
gs://<bucket>/runs/<run_id>/silver/
gs://<bucket>/runs/<run_id>/gold/
```

It also publishes Tableau-ready current-run tables to BigQuery when `gcp.publish_bigquery: true`.
Run-scoped tables are replaced atomically per `run_id` through a staging table and BigQuery
transaction.

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

Use the tracked `configs/portfolio.gcp.yaml` in the container image. Do not deploy an untracked
project-specific config file. Project-specific values should be supplied as Cloud Run environment
variables:

```yaml
gcp:
  enabled: true
  project_id: stock-analysis-prod
  region: us-central1
  bucket: stock-analysis-medallion-prod
  gcs_prefix: runs
  model_registry_prefix: models
  model_artifact_uri: null
  bigquery_location: US
  bigquery_dataset_gold: stock_analysis_gold
  publish_bigquery: true
  allow_model_trained_after_data: false
```

Leave `model_artifact_uri: null` for normal operation. The inference job will read:

```text
gs://<bucket>/models/production/current.json
```

Set `model_artifact_uri` only when you want to pin inference to a specific versioned model, for
example:

```yaml
gcp:
  model_artifact_uri: gs://<bucket>/models/runs/20260503T150000Z/manifest.json
```

Supported deployment overrides:

```text
STOCK_ANALYSIS_GCP_PROJECT_ID
STOCK_ANALYSIS_GCP_REGION
STOCK_ANALYSIS_GCP_BUCKET
STOCK_ANALYSIS_GCP_GCS_PREFIX
STOCK_ANALYSIS_GCP_BIGQUERY_DATASET_GOLD
STOCK_ANALYSIS_GCP_MODEL_REGISTRY_PREFIX
STOCK_ANALYSIS_GCP_MODEL_ARTIFACT_URI
STOCK_ANALYSIS_RUN_ID
STOCK_ANALYSIS_RUN_AS_OF_DATE
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
  --set-env-vars="STOCK_ANALYSIS_GCP_PROJECT_ID=${PROJECT_ID},STOCK_ANALYSIS_GCP_BUCKET=${BUCKET},STOCK_ANALYSIS_GCP_BIGQUERY_DATASET_GOLD=${GOLD_DATASET},STOCK_ANALYSIS_GCP_REGION=${REGION}" \
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
  --set-env-vars="STOCK_ANALYSIS_GCP_PROJECT_ID=${PROJECT_ID},STOCK_ANALYSIS_GCP_BUCKET=${BUCKET},STOCK_ANALYSIS_GCP_BIGQUERY_DATASET_GOLD=${GOLD_DATASET},STOCK_ANALYSIS_GCP_REGION=${REGION}" \
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
gcloud storage cat "gs://${BUCKET}/models/production/current.json"
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
price_coverage
portfolio_recommendations
portfolio_risk_metrics
sector_exposure
run_metadata
forecast_calibration_diagnostics
forecast_calibration_predictions
```

History tables such as `recommendation_lines_history` and `performance_snapshots_history` are not
published by the cloud flow yet because live account tracking is still local/Supabase-backed. Add a
BigQuery-backed account repository before using cloud as the account-performance system of record.

## Tableau

Connect Tableau to Google BigQuery and use:

```text
<project_id>.stock_analysis_gold.portfolio_dashboard_mart
<project_id>.stock_analysis_gold.portfolio_recommendations
<project_id>.stock_analysis_gold.price_coverage
<project_id>.stock_analysis_gold.run_metadata
```

Use `run_id`, `as_of_date`, and `run_data_as_of_date` as dashboard filters.

## Notes

- The local pipeline still writes to `data/runs/<run_id>/`.
- The GCP pipeline writes directly to Cloud Storage through `GcsArtifactStore`.
- GCP model artifacts are stored directly in Cloud Storage through `GcsModelRegistry`.
- Inference loads the promoted GCS model artifact and refuses to retrain implicitly.
- `price_coverage` shows requested-vs-returned ticker coverage, last price date, stale status, and
  whether the ticker had latest feature rows.
- Cloud Scheduler is intentionally not configured in this phase.
- Tableau Hyper export is skipped in the cloud command; BigQuery is the cloud serving layer.
