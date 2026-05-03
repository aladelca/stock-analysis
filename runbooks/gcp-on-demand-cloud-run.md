# GCP On-Demand Cloud Run Job

This runbook deploys the additive GCP pipeline. It does not replace the local `run-one-shot`
workflow.

The cloud command is:

```bash
stock-analysis run-gcp-one-shot --config configs/portfolio.gcp.yaml --forecast-engine ml
```

It writes medallion artifacts directly to Cloud Storage:

```text
gs://<bucket>/runs/<run_id>/raw/
gs://<bucket>/runs/<run_id>/bronze/
gs://<bucket>/runs/<run_id>/silver/
gs://<bucket>/runs/<run_id>/gold/
```

It also publishes Tableau-ready tables to BigQuery when `gcp.publish_bigquery: true`.
Run-scoped tables are appendable and deduplicated by `run_id`; Supabase-derived history tables are
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
  secretmanager.googleapis.com \
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

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

For tighter production IAM, scope BigQuery and Secret Manager permissions to the dataset/secrets
instead of the whole project.

## Configure The Cloud YAML

Edit `configs/portfolio.gcp.yaml`:

```yaml
gcp:
  enabled: true
  project_id: <project_id>
  region: us-central1
  bucket: <bucket>
  gcs_prefix: runs
  bigquery_location: US
  bigquery_dataset_gold: stock_analysis_gold
  publish_bigquery: true
```

If the cloud run should use Supabase account tracking, set:

```yaml
live_account:
  enabled: true
  account_slug: main
  cashflow_source: actual

supabase:
  enabled: true
```

Then create secrets:

```bash
printf '%s' '<supabase_url>' | gcloud secrets create SUPABASE_URL --data-file=-
printf '%s' '<service_role_key>' | gcloud secrets create SUPABASE_SERVICE_ROLE_KEY --data-file=-
```

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

## Create The Cloud Run Job

Without Supabase secrets:

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

With Supabase secrets:

```bash
gcloud run jobs create stock-analysis-one-shot \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}:latest" \
  --region="${REGION}" \
  --service-account="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --memory=8Gi \
  --cpu=4 \
  --task-timeout=3600 \
  --set-secrets="SUPABASE_URL=SUPABASE_URL:latest,SUPABASE_SERVICE_ROLE_KEY=SUPABASE_SERVICE_ROLE_KEY:latest" \
  --args="run-gcp-one-shot,--config,configs/portfolio.gcp.yaml,--forecast-engine,ml"
```

## Execute On Demand

```bash
gcloud run jobs execute stock-analysis-one-shot \
  --region="${REGION}" \
  --wait
```

Inspect logs:

```bash
gcloud run jobs executions list \
  --job=stock-analysis-one-shot \
  --region="${REGION}"
```

List GCS outputs:

```bash
gcloud storage ls "gs://${BUCKET}/runs/"
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
- Cloud Scheduler is intentionally not configured in this phase.
- Tableau Hyper export is skipped in the cloud command; BigQuery is the cloud serving layer.
