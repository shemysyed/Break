# Anonymization Check

Before upload, verify that the ZIP contains none of the following:

- author names, affiliations, email addresses, or identifying repository URLs;
- personal Google Drive, Colab, Linux home, or Windows filesystem paths;
- notebook execution history or account identifiers;
- medical images or direct clinical identifiers;
- authentication tokens, dataset credentials, or Hugging Face tokens;
- comments that reveal author identity or prior submissions.

The exact split CSVs contain only dataset-provided pseudonymous patient/image identifiers needed to reproduce patient grouping. All local source paths were replaced with relative paths or neutral placeholders.

Suggested checks from the package root:

```bash
grep -RniE '(^|[[:space:]"=])(/content/|/home/|[A-Za-z]:\\)|@|hf_[A-Za-z0-9]+' . --exclude=MANIFEST.sha256
```

Review every match manually; programming syntax such as decorators or matrix multiplication can produce benign `@` matches.
