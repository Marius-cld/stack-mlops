/** Vocabulaire de l'interface : libellés des descripteurs et des classes, molécules d'exemple. */

import type { Label, Molecule } from './api/client'

export type Descriptor = keyof Molecule

// Libellés du formulaire, dans l'ordre des colonnes du dataset : `satisfies` fait échouer le build
// si le contrat de l'API ajoute, retire ou renomme un descripteur (après `npm run api:types`)
export const DESCRIPTORS = {
  'SC-5': 'Connectivité de Kier-Hall, cluster simple d’ordre 5',
  'SP-6': 'Connectivité de Kier-Hall, chemin simple d’ordre 6',
  SHBd: 'Somme des E-States des donneurs forts de liaison H',
  minHaaCH: 'E-State minimal des H portés par un CH aromatique',
  maxwHBa: 'E-State maximal des accepteurs faibles de liaison H',
  FMF: 'Part des atomes lourds dans le squelette (complexité)',
} satisfies Record<Descriptor, string>

export const DESCRIPTOR_KEYS = Object.keys(DESCRIPTORS) as Descriptor[]

export const LABEL_TEXT: Record<Label, string> = {
  High_BFE: 'Énergie de liaison élevée',
  Low_BFE: 'Énergie de liaison faible',
}

// Deux molécules du dataset (`data/raw/SIRTUIN6.csv`), une par classe observée, pour essayer le
// modèle : choisies parmi celles qu'il classe nettement bien (~94 % et ~92 % sur la bonne classe)
// plutôt que les premières du fichier, car la ligne 52, premier `Low_BFE`, est prédite `High_BFE`
// à 67 % (une des 19 erreurs sur 100)
export const EXAMPLES: { line: number; observed: Label; molecule: Molecule }[] = [
  {
    line: 2,
    observed: 'High_BFE',
    molecule: {
      'SC-5': 0.540936,
      'SP-6': 7.64192,
      SHBd: 0.162171,
      minHaaCH: 0.44527,
      maxwHBa: 2.20557,
      FMF: 0.467742,
    },
  },
  {
    line: 88,
    observed: 'Low_BFE',
    molecule: {
      'SC-5': 0.242834,
      'SP-6': 2.45381,
      SHBd: 0.589528,
      minHaaCH: 0.416399,
      maxwHBa: 2.12954,
      FMF: 0.225,
    },
  },
]
