#!/bin/bash
# ╔═══════════════════════════════════════════════════════════╗
# ║  Drought Risk Dashboard — Swat Valley, Pakistan           ║
# ╚═══════════════════════════════════════════════════════════╝

echo ""
echo "  🌍  Drought Risk Intelligence Dashboard"
echo "  ──────────────────────────────────────"
echo "  Location : Swat Valley, KPK, Pakistan"
echo "  Coords   : 35.2227°N, 72.4258°E"
echo ""

# Optional: set your OpenWeatherMap API key for live data
# export OWM_API_KEY="your_key_here"

echo "  Starting Flask server on http://localhost:5050"
echo "  Open the URL in your browser to view the dashboard."
echo ""
echo "  API Endpoints:"
echo "    POST http://localhost:5050/predict"
echo "    GET  http://localhost:5050/forecast?months=3"
echo "    GET  http://localhost:5050/live-weather"
echo ""

cd "$(dirname "$0")"
python3 app.py
