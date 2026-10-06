/** Fiche du modèle servi, telle que l'API la publie sur `/v1/model`. */

import { API_URL, type ModelInfo } from '../api/client'
import { metric } from '../format'

function Entries({ title, entries }: { title: string; entries: [string, string][] }) {
  if (entries.length === 0) return null
  return (
    <>
      <h3>{title}</h3>
      <dl className="entries">
        {entries.map(([key, value]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </>
  )
}

/** Affiche provenance MLflow, métriques, hyperparamètres et descripteurs attendus. */
export function ModelCard({ info }: { info: ModelInfo }) {
  const byKey = (a: [string, unknown], b: [string, unknown]) => a[0].localeCompare(b[0])
  const metrics = Object.entries(info.metrics)
    .sort(byKey)
    .map(([key, value]): [string, string] => [key, metric(value)])
  const params = Object.entries(info.params).sort(byKey)

  return (
    <aside className="panel model-card" aria-labelledby="model-title">
      <div className="panel-head">
        <p className="eyebrow">Modèle servi</p>
        <a className="link" href={`${API_URL}/docs`} target="_blank" rel="noreferrer">
          Documentation de l’API ↗
        </a>
      </div>
      <h2 id="model-title">{info.name}</h2>
      <p className="model-version">
        <span className="version">v{info.version}</span>
        <span className="tag">{info.alias}</span>
      </p>
      <dl className="entries">
        <div>
          <dt>Run MLflow</dt>
          <dd className="truncate" title={info.run_id}>
            {info.run_id}
          </dd>
        </div>
        <div>
          <dt>Classe positive</dt>
          <dd>{info.positive_class}</dd>
        </div>
      </dl>

      <Entries title="Métriques" entries={metrics} />
      <Entries title="Hyperparamètres" entries={params} />

      <h3>Descripteurs attendus</h3>
      <ul className="features">
        {info.features.map((feature) => (
          <li key={feature}>
            <code>{feature}</code>
          </li>
        ))}
      </ul>
    </aside>
  )
}
