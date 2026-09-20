# GPR4Net Digital Twin Dashboard

Interactive Streamlit dashboard for the GPR4Net surrogate model, supporting three
predictive-DT scenarios: tunnel-lining inspection at 800 MHz and 600 MHz, and
buried-pipeline detection at 1 GHz.

Deployed on Streamlit Community Cloud and embedded in the project website.

## Local development

```bash
git clone <this-repo-url> gpr4net-dashboard
cd gpr4net-dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Structure

```
.
├── app.py                        # Streamlit application
├── requirements.txt              # Python dependencies
├── models/
│   ├── __init__.py
│   └── d1b1_model.py             # GPR4Net (D1B1Model) architecture
├── checkpoints/
│   ├── tunnel_d1b1.pth           # Tun-800 checkpoint (~33 MB)
│   ├── tunnel_d1a_600mhz.pth     # Tun-600 checkpoint (~33 MB)
│   └── pipe_d2b1.pth             # Pipe checkpoint (~33 MB)
├── assets/
│   └── logo.png                  # Project logo shown in dashboard header
├── .streamlit/
│   └── config.toml               # Theme + server options
└── .gitignore
```

## Deploy to Streamlit Community Cloud

1. Push this repo to a **public** GitHub repository.
2. Open <https://share.streamlit.io> and sign in with GitHub.
3. Click **New app** → pick this repo, branch `main`, main file `app.py`.
4. Click **Deploy**. First deployment builds the environment in about 3–5 minutes.
5. The app is available at `https://<repo-name>-<username>.streamlit.app`.

## Embed into a website (iframe)

Add this HTML anywhere in the project website (works in Markdown, Jekyll, MkDocs,
plain HTML, etc.):

```html
<iframe
  src="https://<repo-name>-<username>.streamlit.app/?embed=true"
  width="100%"
  height="900"
  frameborder="0"
  style="border: 1px solid #ddd; border-radius: 8px;">
</iframe>
```

The `?embed=true` URL parameter tells Streamlit to hide its top toolbar and
footer, so the dashboard looks like an integrated component rather than a
standalone page.

## Reference

Zhu, H., Ye, Z., Wu, H., Giannakis, I., Sheil, B., Ninić, J.
"Building Lightweight Surrogate for GPR Simulation to Enable Predictive Digital
Twin for Underground Infrastructure." Under review, *Tunnelling and Underground
Space Technology*, 2026.

Project: [M-Twin4US](https://github.com/M-Twin4US) — Marie Skłodowska-Curie
Actions grant 101205667.
