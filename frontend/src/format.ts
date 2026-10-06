/** Formats des nombres affichés, à la française (`fr-FR`). */

const percentFormat = new Intl.NumberFormat('fr-FR', {
  style: 'percent',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})
const metricFormat = new Intl.NumberFormat('fr-FR', { maximumFractionDigits: 3 })
// Précision des descripteurs du dataset, comme dans les champs de saisie
const decimalFormat = new Intl.NumberFormat('fr-FR', { maximumFractionDigits: 6 })

export const percent = (value: number) => percentFormat.format(value)
export const metric = (value: number) => metricFormat.format(value)
export const decimal = (value: number) => decimalFormat.format(value)
