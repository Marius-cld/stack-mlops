/** CSV de la prédiction par lot : lecture des molécules, export des résultats. */

import type { Label, Molecule, Prediction } from './api/client'
import { DESCRIPTOR_KEYS } from './molecules'

/**
 * Lit un CSV de molécules : une colonne par descripteur, les autres colonnes sont ignorées.
 * Séparateur `,` ou `;` (export Excel français, virgule décimale).
 */
export function parseMolecules(text: string): Molecule[] {
  const lines = text.replace(/^﻿/, '').split(/\r?\n/)
  const separator = lines[0].includes(';') ? ';' : ','
  const cells = (line: string) =>
    line.split(separator).map((cell) => cell.trim().replace(/^"(.*)"$/, '$1'))

  const header = cells(lines[0])
  const missing = DESCRIPTOR_KEYS.filter((key) => !header.includes(key))
  if (missing.length > 0) throw new Error(`colonnes manquantes : ${missing.join(', ')}`)

  const molecules: Molecule[] = []
  lines.forEach((line, index) => {
    if (index === 0 || line.trim() === '') return
    const row = cells(line)
    const entries = DESCRIPTOR_KEYS.map((key) => {
      const raw = row[header.indexOf(key)] ?? ''
      const value = Number(separator === ';' ? raw.replace(',', '.') : raw)
      // `Number('')` vaut 0 : une cellule vide est une valeur manquante, pas un zéro
      if (raw === '' || !Number.isFinite(value)) {
        throw new Error(`ligne ${index + 1}, ${key} : « ${raw} » n’est pas un nombre`)
      }
      return [key, value]
    })
    molecules.push(Object.fromEntries(entries) as Molecule)
  })
  if (molecules.length === 0) throw new Error('aucune molécule après la ligne d’en-tête')
  return molecules
}

/** Écrit le CSV des résultats : descripteurs, classe prédite et probabilité par classe. */
export function toCsv(molecules: Molecule[], predictions: Prediction[], classes: Label[]): string {
  const header = [...DESCRIPTOR_KEYS, 'label', ...classes.map((label) => `proba_${label}`)]
  const rows = molecules.map((molecule, i) => [
    ...DESCRIPTOR_KEYS.map((key) => molecule[key]),
    predictions[i].label,
    ...classes.map((label) => predictions[i].probabilities[label]),
  ])
  return [header, ...rows].map((row) => row.join(',')).join('\n') + '\n'
}
