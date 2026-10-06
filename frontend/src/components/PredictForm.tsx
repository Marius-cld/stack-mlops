/** Onglet « Une molécule » : saisie des six descripteurs et prédiction unitaire. */

import { useState, type FormEvent } from 'react'
import {
  errorMessage,
  predict,
  type ModelInfo,
  type Molecule,
  type PredictionResponse,
} from '../api/client'
import { DESCRIPTORS, DESCRIPTOR_KEYS, EXAMPLES, type Descriptor } from '../molecules'
import { PredictionResult } from './Prediction'

// Saisies gardées en texte : un champ en cours d'édition peut être vide ou incomplet
type Values = Record<Descriptor, string>

const toValues = (molecule: Molecule) =>
  Object.fromEntries(DESCRIPTOR_KEYS.map((key) => [key, String(molecule[key])])) as Values

const toMolecule = (values: Values) =>
  Object.fromEntries(DESCRIPTOR_KEYS.map((key) => [key, Number(values[key])])) as Molecule

/** Prédit la molécule saisie ; le résultat est marqué périmé dès qu'une valeur change. */
export function PredictForm({ info }: { info: ModelInfo }) {
  const [values, setValues] = useState(() => toValues(EXAMPLES[0].molecule))
  const [result, setResult] = useState<{ response: PredictionResponse; values: Values } | null>(
    null,
  )
  const [error, setError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)
  const stale = result !== null && DESCRIPTOR_KEYS.some((key) => result.values[key] !== values[key])

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setPending(true)
    setError(null)
    try {
      setResult({ response: await predict(toMolecule(values)), values })
    } catch (err) {
      setResult(null)
      setError(errorMessage(err))
    } finally {
      setPending(false)
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="examples">
        <span>Remplir avec :</span>
        {EXAMPLES.map((example) => (
          <button
            key={example.line}
            type="button"
            className="chip"
            title={`Ligne ${example.line} de SIRTUIN6.csv, classe observée ${example.observed}`}
            onClick={() => setValues(toValues(example.molecule))}
          >
            Exemple {example.observed}
          </button>
        ))}
      </div>

      <div className="fields">
        {DESCRIPTOR_KEYS.map((key) => (
          <label key={key} className="field">
            <span className="field-name">{key}</span>
            <span className="field-help">{DESCRIPTORS[key]}</span>
            <input
              type="number"
              step="any"
              required
              value={values[key]}
              onChange={(event) => {
                const value = event.target.value
                setValues((current) => ({ ...current, [key]: value }))
              }}
            />
          </label>
        ))}
      </div>

      <div className="actions">
        <button type="submit" className="button primary" disabled={pending}>
          {pending ? 'Prédiction…' : 'Prédire'}
        </button>
        {stale && <span className="hint">Valeurs modifiées depuis cette prédiction</span>}
      </div>

      {error && (
        <p className="alert" role="alert">
          {error}
        </p>
      )}
      <div aria-live="polite">
        {result && (
          <PredictionResult
            prediction={result.response}
            model={result.response.model}
            info={info}
            stale={stale}
          />
        )}
      </div>
    </form>
  )
}
