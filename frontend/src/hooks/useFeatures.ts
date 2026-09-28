import { useQuery } from '@tanstack/react-query'
import { featuresApi } from '@/lib/api'

/**
 * Server feature flags from `GET /features`. Fetched once per QueryClient
 * (`staleTime: Infinity`): flags change only on a redeploy. While loading,
 * `data` is undefined, so callers treat a flag as off until it is known on.
 */
export function useFeatures() {
  return useQuery({
    queryKey: ['features'],
    queryFn: () => featuresApi.get(),
    staleTime: Infinity,
  })
}
