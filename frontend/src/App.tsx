/** Application : état de l'API, onglets de prédiction (molécule ou lot CSV), fiche du modèle. */

import { useEffect, useState } from 'react'
import { API_URL, errorMessage, getModelInfo, type ModelInfo } from './api/client'
import { BatchPredict } from './components/BatchPredict'
import { ModelCard } from './components/ModelCard'
import { PredictForm } from './components/PredictForm'
import './App.css'

type ApiState =
  | { status: 'loading' }
  | { status: 'ready'; info: ModelInfo }
  | { status: 'offline'; message: string }

/** Charge la fiche du modèle servi, dont dépend toute l'interface ; `retry` la recharge. */
function useModelInfo() {
  const [state, setState] = useState<ApiState>({ status: 'loading' })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let ignore = false
    getModelInfo().then(
      (info) => {
        if (!ignore) setState({ status: 'ready', info })
      },
      (error) => {
        if (!ignore) setState({ status: 'offline', message: errorMessage(error) })
      },
    )
    return () => {
      ignore = true
    }
  }, [attempt])

  const retry = () => {
    setState({ status: 'loading' })
    setAttempt((n) => n + 1)
  }
  return [state, retry] as const
}

function ApiStatus({ state }: { state: ApiState }) {
  if (state.status === 'loading') {
    return <span className="status">Connexion à l’API…</span>
  }
  if (state.status === 'offline') {
    return <span className="status status-offline">API injoignable</span>
  }
  const { name, version, alias } = state.info
  return (
    <span className="status status-ready">
      {name} v{version} · {alias}
    </span>
  )
}

function Offline({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <section className="panel offline" role="alert">
      <h2>L’API de serving ne répond pas</h2>
      <p className="alert">{message}</p>
      <p>
        Elle doit tourner sur <code>{API_URL}</code> : <code>make up</code> (conteneur) ou{' '}
        <code>make run</code> (local) à la racine de la stack. Elle refuse de démarrer sans
        registre MLflow joignable ni version sous l’alias servi.
      </p>
      <button type="button" className="button primary" onClick={onRetry}>
        Réessayer
      </button>
    </section>
  )
}

const TABS = [
  { id: 'single', title: 'Une molécule' },
  { id: 'batch', title: 'Lot CSV' },
] as const

/** Affiche les onglets de prédiction à côté de la fiche du modèle. */
function Workspace({ info }: { info: ModelInfo }) {
  const [tab, setTab] = useState<(typeof TABS)[number]['id']>('single')

  return (
    <div className="layout">
      <section className="panel">
        <div className="panel-head tabs" role="tablist" aria-label="Mode de prédiction">
          {TABS.map(({ id, title }) => (
            <button
              key={id}
              id={`tab-${id}`}
              type="button"
              role="tab"
              className="tab"
              aria-selected={tab === id}
              aria-controls={`panel-${id}`}
              onClick={() => setTab(id)}
            >
              {title}
            </button>
          ))}
        </div>
        {/* Les deux onglets restent montés : changer d'onglet ne perd ni saisie ni résultats */}
        <div id="panel-single" role="tabpanel" aria-labelledby="tab-single" hidden={tab !== 'single'}>
          <PredictForm info={info} />
        </div>
        <div id="panel-batch" role="tabpanel" aria-labelledby="tab-batch" hidden={tab !== 'batch'}>
          <BatchPredict info={info} />
        </div>
      </section>
      <ModelCard info={info} />
    </div>
  )
}

function App() {
  const [state, retry] = useModelInfo()

  return (
    <>
      <header className="masthead">
        <div className="container">
          <div className="topbar">
            <span className="brand">
              {/* Même fichier que le favicon (`index.html`) */}
              <img src={`${import.meta.env.BASE_URL}images.svg`} alt="" width={28} height={28} />
              Sirtuin6 BFE
            </span>
            <ApiStatus state={state} />
          </div>
          <hgroup>
            <h1>Énergie de liaison sur Sirtuin 6</h1>
            <p>Enzyme codée par un gène situé sur le chromosome 19 humain</p>
          </hgroup>
          <p>
            Classez une petite molécule en énergie de liaison élevée (<code>High_BFE</code>) ou
            faible (<code>Low_BFE</code>) à partir de six descripteurs PaDEL, avec le modèle servi
            par l’API.
          </p>
        </div>
      </header>

      <main className="container page">
        {state.status === 'loading' && <p className="muted">Chargement du modèle servi…</p>}
        {state.status === 'offline' && <Offline message={state.message} onRetry={retry} />}
        {state.status === 'ready' && <Workspace info={state.info} />}
      </main>

      <footer className="container footer">
        API <code>{API_URL}</code> ·{' '}
        <a className="link" href={`${API_URL}/openapi.json`} target="_blank" rel="noreferrer">
          contrat OpenAPI
        </a>
      </footer>
    </>
  )
}

export default App
