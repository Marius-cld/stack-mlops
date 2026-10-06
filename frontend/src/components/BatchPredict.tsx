/** Onglet « Lot CSV » : prédiction d'un fichier de molécules en un appel. */

import { useState, type DragEvent } from 'react'
import {
  errorMessage,
  predictBatch,
  type BatchResponse,
  type ModelInfo,
  type Molecule,
} from '../api/client'
import { parseMolecules, toCsv } from '../csv'
import { decimal, percent } from '../format'
import { DESCRIPTOR_KEYS } from '../molecules'
import { LabelBadge, ModelStamp, ProbabilityBar } from './Prediction'

type Batch = { fileName: string; molecules: Molecule[] }

/** Fait télécharger `content` par le navigateur, sous le nom `fileName`. */
function download(fileName: string, content: string) {
  const url = URL.createObjectURL(new Blob([content], { type: 'text/csv' }))
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  link.click()
  URL.revokeObjectURL(url)
}

/** Lit un CSV déposé ou choisi, prédit ses molécules et propose les résultats en CSV. */
export function BatchPredict({ info }: { info: ModelInfo }) {
  const [batch, setBatch] = useState<Batch | null>(null)
  const [result, setResult] = useState<BatchResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)
  const [dragging, setDragging] = useState(false)
  const positive = info.positive_class

  async function load(file: File) {
    setBatch(null)
    setResult(null)
    setError(null)
    try {
      setBatch({ fileName: file.name, molecules: parseMolecules(await file.text()) })
    } catch (err) {
      setError(`${file.name} : ${errorMessage(err)}`)
    }
  }

  async function handlePredict() {
    if (!batch) return
    setPending(true)
    setError(null)
    try {
      setResult(await predictBatch(batch.molecules))
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setPending(false)
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setDragging(false)
    const file = event.dataTransfer.files[0]
    if (file && !pending) load(file)
  }

  return (
    <div className="batch">
      <div
        className={`dropzone${dragging ? ' dragging' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false)
        }}
        onDrop={handleDrop}
      >
        <p className="dropzone-title">Déposez un fichier CSV ici, ou choisissez-le</p>
        <p className="muted">
          Une ligne d’en-tête avec les colonnes{' '}
          {DESCRIPTOR_KEYS.map((key, i) => (
            <span key={key}>
              {i > 0 && ', '}
              <code>{key}</code>
            </span>
          ))}
          , puis une molécule par ligne. Les autres colonnes sont ignorées ; séparateur{' '}
          <code>,</code> ou <code>;</code>.
        </p>
        <div className="actions">
          <label className="button secondary">
            Choisir un fichier
            <input
              type="file"
              accept=".csv,text/csv"
              className="visually-hidden"
              disabled={pending}
              onChange={(event) => {
                const file = event.target.files?.[0]
                // Vidé pour pouvoir recharger le même fichier après l'avoir corrigé
                event.target.value = ''
                if (file) load(file)
              }}
            />
          </label>
          <button
            type="button"
            className="button primary"
            disabled={!batch || pending}
            onClick={handlePredict}
          >
            {pending
              ? 'Prédiction…'
              : batch
                ? `Prédire ${batch.molecules.length} molécules`
                : 'Prédire le lot'}
          </button>
        </div>
        {batch && (
          <p className="file">
            <code>{batch.fileName}</code> · {batch.molecules.length} molécules lues
          </p>
        )}
      </div>

      {error && (
        <p className="alert" role="alert">
          {error}
        </p>
      )}

      {result && batch && (
        <section className="batch-result" aria-label="Résultats du lot" aria-live="polite">
          <div className="batch-summary">
            <div className="counts">
              {info.classes.map((label) => (
                <span key={label}>
                  <LabelBadge label={label} positive={positive} />
                  <strong>{result.predictions.filter((p) => p.label === label).length}</strong>
                </span>
              ))}
            </div>
            <button
              type="button"
              className="button secondary"
              onClick={() =>
                download(
                  `${batch.fileName.replace(/\.csv$/i, '')}_predictions_v${result.model.version}.csv`,
                  toCsv(batch.molecules, result.predictions, info.classes),
                )
              }
            >
              Télécharger le CSV
            </button>
          </div>
          <ModelStamp model={result.model} />

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th className="num">#</th>
                  <th>Classe prédite</th>
                  <th>P({positive})</th>
                  {DESCRIPTOR_KEYS.map((key) => (
                    <th key={key} className="num">
                      {key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.predictions.map((prediction, i) => (
                  <tr key={i}>
                    <td className="num muted">{i + 1}</td>
                    <td>
                      <LabelBadge label={prediction.label} positive={positive} />
                    </td>
                    <td>
                      <span className="mini">
                        <ProbabilityBar value={prediction.probabilities[positive]} />
                        <span className="num">{percent(prediction.probabilities[positive])}</span>
                      </span>
                    </td>
                    {DESCRIPTOR_KEYS.map((key) => (
                      <td key={key} className="num">
                        {decimal(batch.molecules[i][key])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  )
}
