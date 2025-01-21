import { useEffect } from 'react';
import { useRouter } from 'next/router';

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to comprehensive dashboard
    router.push('/comprehensive');
  }, [router]);

  return null; // No need to render anything as we're redirecting
}
