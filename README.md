# SMS Spam Detection & Prediction Service

Machine learning pipeline and FastAPI microservice for classifying SMS messages as legitimate (`ham`) or unsolicited (`spam`).

## Dataset

The model is trained on the SMS Spam Collection dataset (`SMSSpamCollection.csv`), consisting of 5,574 labeled messages.

### Schema
| Column | Type | Description |
|---|---|---|
| `label` | string | Target classification (`ham` or `spam`) |
| `message` | string | Raw text body of the SMS message |

### Class Distribution
- **Ham (legitimate):** 4,825 instances (~86.56%)
- **Spam (unsolicited):** 749 instances (~13.44%)
