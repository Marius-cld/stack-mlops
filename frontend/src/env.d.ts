/** Variables d'environnement `VITE_*` exposées au code par Vite (`import.meta.env`). */
interface ImportMetaEnv {
  /** URL de l'API de serving, `http://localhost:8001` par défaut (voir `.env.example`). */
  readonly VITE_API_URL?: string
}
