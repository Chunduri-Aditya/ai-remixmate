import { normalizeKey } from '../CamelotWheel'

describe('normalizeKey', () => {
  it('strips " major"',        () => expect(normalizeKey('C major')).toBe('C'))
  it('appends m for minor',    () => expect(normalizeKey('A minor')).toBe('Am'))
  it('handles already-normalized', () => expect(normalizeKey('Am')).toBe('Am'))
  it('handles F# major',       () => expect(normalizeKey('F# major')).toBe('F#'))
  it('returns input unchanged when unknown', () => expect(normalizeKey('Xyz')).toBe('Xyz'))
  it('returns undefined for undefined input', () => expect(normalizeKey(undefined)).toBeUndefined())
})
