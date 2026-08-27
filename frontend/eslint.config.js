// Flat config (ESLint 9). Replaces the .eslintrc format that `npm run lint` expected
// but never had — the script had been failing with "couldn't find a config file".
import js from '@eslint/js'
import tsParser from '@typescript-eslint/parser'
import tsPlugin from '@typescript-eslint/eslint-plugin'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'

export default [
  { ignores: ['dist/**', 'coverage/**', 'node_modules/**'] },

  js.configs.recommended,

  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      parser: tsParser,
      ecmaVersion: 2022,
      sourceType: 'module',
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },
    plugins: {
      '@typescript-eslint': tsPlugin,
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...tsPlugin.configs.recommended.rules,
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],

      // tsc already enforces these with noUnusedLocals/noUnusedParameters; the
      // ESLint copy only duplicates the error, so defer to the compiler.
      '@typescript-eslint/no-unused-vars': 'off',

      // TypeScript resolves identifiers; no-undef reports false positives on
      // type-only names and DOM globals under the TS parser.
      'no-undef': 'off',
    },
  },
]
