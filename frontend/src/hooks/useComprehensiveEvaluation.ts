import { useEffect, useState } from 'react';
import { API_CONFIG } from '../lib/config';

interface ComprehensiveEvaluationData {
  indicators: {
    stability: Record<string, boolean>;
    maneuvers: Record<string, boolean>;
    rf_indicators: Record<string, boolean>;
    physical_characteristics: Record<string, boolean>;
    orbital_characteristics: Record<string, boolean>;
    launch_indicators: Record<string, boolean>;
    compliance: Record<string, boolean>;
  };
  metadata: {
    active_indicators: number;
    total_indicators: number;
    threat_score: number;
    evaluation_timestamp: string;
  };
}

export const useComprehensiveEvaluation = (objectId: string) => {
  const [data, setData] = useState<ComprehensiveEvaluationData | undefined>();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | undefined>();

  useEffect(() => {
    const fetchData = async () => {
      if (!objectId) return;

      setIsLoading(true);
      setError(undefined);

      try {
        const response = await fetch(`${API_CONFIG.baseUrl}/evaluate/comprehensive`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            data: { object_id: objectId }
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to fetch comprehensive evaluation');
        }

        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [objectId]);

  return { data, isLoading, error };
}; 