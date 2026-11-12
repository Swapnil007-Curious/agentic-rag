---
title: Agentic-RAG Study Assistant
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
license: mit
pinned: false
short_description: Chat assistant grounded in open textbooks and Indian law
---

# Agentic-RAG Study Assistant

A chat assistant for students and developers: concepts, code, math and science, and questions
about five document collections (computer science and AI textbooks, Indian consumer law, Indian
constitutional, tax and criminal law, human biology, physics and chemistry). You can also upload
up to three PDFs of your own.

* Every message is answered by one streamed call to the hosted model (openai/gpt-oss-120b on
  Groq). When the message matches the selected collection or your uploads, the best passages are
  added automatically and the reply cites them as clickable chips; otherwise it answers from
  general knowledge and says so.
* **Verify** checks a cited reply with deterministic citation and number guards and a critic.
* **Grounded only** answers strictly from the documents or says they do not cover the question.

## Privacy

Messages and the matching passages are sent to the model provider (Groq) to produce the answer.
The server does not log message contents. Conversations are stored only in your browser.
Uploaded PDFs are processed in memory, never written to disk or logs, and deleted when you close
the page or after 2 hours of inactivity.

## Limits

This free demo shares one API quota: 20 messages per 10 minutes and 200 per day per visitor,
messages of at most 4000 characters, and a global daily request cap. When the provider is busy
the app says so; try again in a minute.

## Licences and non-commercial use

The application code is MIT-licensed. The document excerpts keep their own licences, listed in
the app under "About & sources" and in each collection's SOURCES.csv: among them CC BY-SA (Dive
into Deep Learning), CC BY-NC-SA (Pro Git, the OpenStax titles as recorded in their PDFs),
CC BY-NC (Think Python), CC BY (Open Data Structures), the PSF licence (the Python Tutorial) and
Indian government publications (statutes). Because several are non-commercial, this demo is
free and non-commercial. Source PDFs are not included; only the text chunks and their
embeddings used for search are.

Answers can be wrong. Law collections: not legal advice. Human biology: educational, not medical
advice; for symptoms or emergencies contact a doctor or local emergency services.

## Configuration

* Space secret `LLM_API_KEY`: the hosted provider's API key (required).
* The image binds `0.0.0.0:7860` (`HOST`, `PORT`) and downloads bge-small at build time.
