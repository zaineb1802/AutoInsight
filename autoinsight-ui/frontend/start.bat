@echo off
echo Starting AutoInsight frontend...
cd /d "%~dp0"
npm install
npm run dev
