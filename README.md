---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://huggingface.co/spaces/Lucas-C-R/Agents
cd Agents
```

2. Create and activate a virtual environment:
- Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

 - Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

1. Install dependencies:
```bash
pip install -r requirements.txt
```

1. Configure environment variables:
```bash
# API Keys
export OPENAI_API_KEY="sk-proj-..."  # Your OpenAI API key
export TAVILY_API_KEY="tvly-dev-..." # Your Tavily API key
export GROQ_API_KEY="gsk_..."        # Your Groq API key
export HF_TOKEN="hf_..."             # Your Hugging Face token

# Space Configuration
export SPACE_ID="<your user>/<space name>"  # Your Hugging Face space ID

# Supabase Configuration
export SUPABASE_URL="your_supabase_url"     # Your Supabase project URL
export SUPABASE_KEY="your_supabase_key"     # Your Supabase API key
```

5. Run the application:
```bash
python app.py
```

## 🔧 Detailed Configuration

### 1. Supabase Setup

1. Create an organization on [Supabase](https://supabase.com/)
2. Create a new project
3. Set up a password and select the nearest region
4. In the "Project Overview" tab, locate the "Project API" section
5. Copy the Project URL to the `SUPABASE_URL` environment variable
6. Copy the API Key to the `SUPABASE_KEY` environment variable

### 2. Database Configuration

Execute the following SQL in the Supabase SQL editor:

```sql
create extension if not exists vector;
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  content text,
  metadata jsonb,
  embedding vector(768)
);

DROP FUNCTION IF EXISTS match_documents_langchain(vector, integer);

create or replace function match_documents_langchain(
    query_embedding vector(768),
    match_count int default 5
)
returns table (
    id uuid,
    content text,
    metadata jsonb,
    similarity float
)
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$ language plpgsql;
```

### 3. Data Import

1. Access the "Table Editor" tab in Supabase
2. Select the "documents" table
3. Click on "Import data from CSV"
4. Select the `supabase_docs.csv` file from the project directory
5. Click "Import data" and wait for completion

## 📝 License

This project is under the MIT License. See the LICENSE file for more details.