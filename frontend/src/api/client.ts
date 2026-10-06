/**
 * Client de l'API de serving.
 *
 * Les types viennent du contrat publié par l'API (`schema.d.ts`, généré par `npm run api:types`) :
 * le frontend ne connaît de l'API que son URL et ce contrat.
 */

import type { components } from './schema'

type Schemas = components['schemas']
export type Molecule = Schemas['Molecule']
export type Label = Schemas['Prediction']['label']
export type Prediction = Schemas['Prediction']
export type ModelRef = Schemas['ModelRef']
export type ModelInfo = Schemas['ModelInfo']
export type PredictionResponse = Schemas['PredictionResponse']
export type BatchResponse = Schemas['BatchResponse']

export const API_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8001').replace(/\/$/, '')

export class ApiError extends Error {}

export const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error)

/** Appelle l'API (GET sans `body`, POST JSON sinon) ; lève `ApiError` avec un message lisible. */
async function request<T>(path: string, body?: unknown): Promise<T> {
  let response: Response
  try {
    response = await fetch(
      `${API_URL}${path}`,
      body === undefined
        ? undefined
        : {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
          },
    )
  } catch {
    // Le navigateur ne distingue pas une API arrêtée d'une origine refusée par CORS
    throw new ApiError(`API injoignable sur ${API_URL} (arrêtée, ou origine refusée par CORS)`)
  }
  if (!response.ok) throw new ApiError(await describeError(response))
  return (await response.json()) as T
}

/** Résume une erreur ; un 422 donne une ligne par champ refusé, sans le préfixe `body`. */
async function describeError(response: Response): Promise<string> {
  const body = await response.json().catch(() => null)
  if (response.status === 422 && Array.isArray(body?.detail)) {
    return (body.detail as Schemas['ValidationError'][])
      .map((error) => `${error.loc.slice(1).join(' › ')} : ${error.msg}`)
      .join('\n')
  }
  return `L'API a répondu ${response.status} ${response.statusText}`
}

export const getModelInfo = () => request<ModelInfo>('/v1/model')

export const predict = (molecule: Molecule) =>
  request<PredictionResponse>('/v1/predict', molecule)

export const predictBatch = (molecules: Molecule[]) =>
  request<BatchResponse>('/v1/predict/batch', { molecules } satisfies Schemas['BatchRequest'])
