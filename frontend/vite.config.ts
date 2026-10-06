/** Configuration de Vite (https://vite.dev/config/) : serveur de dev et prévisualisation. */

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Seule origine autorisée par défaut par le CORS de l'API (`FRONTEND_PORT`, `config/settings.py`) :
// échouer si le port est pris plutôt que basculer sur un port que l'API refuserait
const port = 5173

export default defineConfig({
  plugins: [react()],
  server: { port, strictPort: true },
  preview: { port, strictPort: true },
})
