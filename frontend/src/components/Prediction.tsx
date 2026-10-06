/** Affichage des prédictions, partagé par les deux onglets : classe, probabilités, modèle. */

import { Fragment } from 'react'
import type { Label, ModelInfo, ModelRef, Prediction } from '../api/client'
import { percent } from '../format'
import { LABEL_TEXT } from '../molecules'

export function LabelBadge({ label, positive }: { label: Label; positive: Label }) {
  return (
    <span className={`badge ${label === positive ? 'badge-positive' : 'badge-negative'}`}>
      {label}
    </span>
  )
}

/** Trace une probabilité en barre, décorative : le pourcentage est toujours écrit à côté. */
export function ProbabilityBar({ value, muted = false }: { value: number; muted?: boolean }) {
  return (
    <span className="bar" aria-hidden="true">
      <span className={`bar-fill${muted ? ' muted' : ''}`} style={{ width: `${value * 100}%` }} />
    </span>
  )
}

/** Affiche la version du registre MLflow qui a produit la réponse (traçabilité). */
export function ModelStamp({ model }: { model: ModelRef }) {
  return (
    <p className="stamp">
      Prédit par <code>{model.name}</code> v{model.version} · alias <code>{model.alias}</code>
    </p>
  )
}

/** Affiche la classe prédite et la probabilité de chaque classe ; `stale` estompe le résultat. */
export function PredictionResult({
  prediction,
  model,
  info,
  stale,
}: {
  prediction: Prediction
  model: ModelRef
  info: ModelInfo
  stale: boolean
}) {
  return (
    <section className={`result${stale ? ' stale' : ''}`} aria-label="Résultat de la prédiction">
      <div className="result-head">
        <p className="result-label">{LABEL_TEXT[prediction.label]}</p>
        <LabelBadge label={prediction.label} positive={info.positive_class} />
      </div>
      <div className="probas">
        {info.classes.map((label) => (
          <Fragment key={label}>
            <code>{label}</code>
            <ProbabilityBar
              value={prediction.probabilities[label]}
              muted={label !== prediction.label}
            />
            <span className="num">{percent(prediction.probabilities[label])}</span>
          </Fragment>
        ))}
      </div>
      <ModelStamp model={model} />
    </section>
  )
}
