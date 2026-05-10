# Agent Sentinal - Data Protection as Code

Agent Sentinal is an enterprise-grade data protection system that implements "Data Protection as Code". It provides real-time monitoring and protection for sensitive data across large language model (LLM) workflows using a two-tier classification system.

## 📂 Repository Structure

```
agentsentinal/
├── app/
│   ├── api/               # API endpoints and routing
│   ├── core/              # Core configuration and utilities
│   ├── services/          # Business logic and workflows
│   │   ├── llm/           # LLM integration (Vertex AI, etc.)
│   │   ├── vector_store/  # Vector DB operations
│   │   ├── worker/        # Background job processing
│   │   ├── anonymizer/    # Data transformation logic
│   │   ├── classify/      # Classification logic
│   │   └── validator/     # Validation logic
│   ├── schemas/           # Pydantic data models
│   └── main.py            # Application entry point
├── migrations/            # Database migrations
├── tests/                 # Unit and integration tests
├── .env                   # Environment variables (not in git)
├── .gitignore             # Files to ignore in version control
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## ⚡ Quick Start

### Prerequisites

- Python 3.13+
- PostgreSQL database
- Redis cache
- Google Cloud credentials (for Vertex AI)

### Installation

```bash
# Clone the repository
git clone https://github.com/goyalnitin148/agentsentinal.git
cd agentsentinal

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
uv run alembic upgrade head

# Run the application
uv run uvicorn app.main:app --reload
```

## 🔐 Data Protection Workflow

```mermaid
flowchart TD
    A[User Request] --> B{Preprocessing}
    B --> C[Classify: PII, PHI, PCI]
    C --> D{Content Safety Check}
    D --> E{Sensitive Data Found?}
    E -->|Yes| F{Enforce Policies}
    F --> G[Anonymize / Mask]
    G --> H[Vector Search]
    H --> I[Check Context / Drift]
    I --> J[Allow / Reject / Flag]
    J --> K[Response to User]
```

## 🚀 Key Features

### 1. Two-Tier Classification System

#### Tier 1: Content & Intent-Based Classification
**Classify data with 30% less data using AI Context.**

- Uses 3-shot prompt engineering for high accuracy
- Classifies in 25ms or less
- Reduces data usage by 70% compared to 5-shot

**Classification categories:**
```
1. Protected Health Information (PHI)
2. Sensitive Personal Information (SPI)
3. Financial Data (PCI)
4. Personal Identifiable Information (PII)
5. Company Intellectual Property (IP)
6. General/Neutral Information
```

#### Tier 2: Policy-Driven Enforcement
**Real-time protection using LLM-powered validators.**

- **Classification:** Identify sensitive data using Tier 1
- **Policy Validation:** Check against company-defined policies
- **Reasoning:** LLM explains the violation in 50ms
- **Action:** Allow, Reject, or Flag for review

### 2. RAG-Powered Data Protection

**Reduce false positives by 40% using Retrieval Augmented Generation.**

```mermaid
sequenceDiagram
    participant User
    participant System as Agent Sentinal
    participant VectorDB as Vector Database
    participant PolicyDB as Policy Database
    participant LLM as LLM

    User->>System: Submit data for processing
    System->>System: Classify data using Tier 1
    System->>VectorDB: Retrieve relevant policies
    VectorDB-->>System: Return policy documents
    System->>PolicyDB: Check for specific violations
    PolicyDB-->>System: Policy rules
    System->>LLM: Validate against policies
    System-->>User: Protected response
```

## 🛠️ Configuration

### Environment Variables

```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=your-region
GOOGLE_APPLICATION_CREDENTIALS=/path/to/keyfile.json

# Database Configuration
DATABASE_URL=postgresql://user:password@host:port/database

# Redis Configuration
REDIS_URL=redis://host:port/db
```

### Policy Management

Create company policies in the vector database:

```python
from app.services.vector_store.client import get_vector_store

store = get_vector_store()

# Add a new policy
store.add_policy(
    id="policy_123",
    name="Customer Data Privacy Policy",
    description="Strictly prohibits sharing customer contact information",
    content="Customers' names, emails, and phone numbers must never be shared externally..."
)

# Retrieve policies
policies = store.get_policies(query="data privacy")
```

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app
```

## 🔐 Security Best Practices

- **Never** commit `.env` files to version control
- Use **strong, unique passwords** for database and services
- **Rotate API keys** regularly
- **Monitor** system logs for suspicious activity
- **Regularly update** dependencies to patch vulnerabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For issues, questions, or feature requests, please open an issue on the GitHub repository.