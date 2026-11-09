# Deploying to AWS

This puts the scoring API online so anyone can try it in a browser. It uses three AWS services and no Docker:

| Service | Job |
|---|---|
| **S3** | Stores each trained model version |
| **RDS for PostgreSQL** | Stores every scored application and decision |
| **Elastic Beanstalk** | Runs the API on a small server and restarts it if it fails |

```mermaid
flowchart LR
    U[Browser or app] -->|POST /score| EB[Elastic Beanstalk<br/>FastAPI server]
    EB -->|downloads model at startup| S3[(S3<br/>model versions)]
    EB -->|saves each decision| RDS[(RDS PostgreSQL<br/>applications + reasons)]
    M[monitor.py --db] -->|SQL| RDS
```

**Cost:** new AWS accounts get free credits for their first months. Check the current terms at [aws.amazon.com/free](https://aws.amazon.com/free), set a budget alert (step 1), and shut everything down when you're done (last step).

## 1. Account and budget
1. Create an AWS account at [aws.amazon.com](https://aws.amazon.com).
2. In the console, open **Billing → Budgets** and create a monthly budget of $5 with an email alert.
3. In **IAM**, create a user for yourself with programmatic access and create an access key for it.

## 2. Command-line tools

```bash
winget install -e --id Amazon.AWSCLI
```

```bash
pip install awsebcli
```

Then enter your access key yourself when prompted. Use region `us-east-1`:

```bash
aws configure
```

## 3. Upload the model to S3
Bucket names are global, so add something unique to the name:

```bash
aws s3 mb s3://noah-credit-model-2026
```

```bash
python src/upload_model.py s3://noah-credit-model-2026/models/
```

It prints the `MODEL_S3_URI` to use in step 6.

## 4. Create the database (RDS)
In the console, open **RDS → Create database**:
- Engine: **PostgreSQL**. Template: **Free tier**. Instance: `db.t4g.micro`, 20 GB storage.
- Set a master username and password, and keep them somewhere safe.
- Public access: **No**.
- Under **Additional configuration**, set the initial database name to `loans`.

When it's ready, copy its **endpoint**. The API creates its tables on first start.

## 5. Create the server (Elastic Beanstalk)
Run these from this folder, after the code is committed to Git:

```bash
eb init -p python-3.11 credit-default-api --region us-east-1
```

```bash
eb create credit-default-api --single --instance_type t3.micro
```

Then give the server access to the other two services:
- **S3:** in **IAM → Roles**, open `aws-elasticbeanstalk-ec2-role` and add a policy that allows `s3:GetObject` on `arn:aws:s3:::noah-credit-model-2026/models/*`.
- **Database:** in **EC2 → Security Groups**, open the database's security group. Add an inbound rule for PostgreSQL (port 5432) whose source is the Elastic Beanstalk instance's security group.

## 6. Connect the pieces
Type this yourself with your real values. Never put the password in a file or a commit:

```bash
eb setenv MODEL_S3_URI=s3://noah-credit-model-2026/models/credit-default-1.0.0.joblib DATABASE_URL=postgresql://USER:PASSWORD@ENDPOINT:5432/loans
```

## 7. Try it

```bash
eb open
```

Add `/docs` to the address. FastAPI's built-in page lets anyone fill in an application and score it, which makes it the link to share. `/health` should report `"database": "postgresql"`.

To check drift on what the live API has scored, run this locally with the same `DATABASE_URL`. It works once your computer is allowed to reach the database:

```bash
python src/monitor.py --db --days 30
```

## Updating and rolling back
- **New code:** commit it, then run `eb deploy`.
- **New model:** retrain with a new version number, run `upload_model.py`, then point `MODEL_S3_URI` at the new file with `eb setenv`. To roll back, point it at the old file.

## Shutting down

```bash
eb terminate credit-default-api
```

Then delete the RDS database in the console. Choose to skip the final snapshot unless you want to keep the data.

## Known limits
- The default Elastic Beanstalk address uses HTTP, not HTTPS. Adding HTTPS needs a custom domain and a certificate.
- The API has no login, so anyone with the link can score applications and add rows to the database. That's fine for a demo, but a real lender would add authentication and rate limits.
