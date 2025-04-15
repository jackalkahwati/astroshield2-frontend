import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  BarChart3,
  Boxes,
  CircleAlert,
  Home,
  Rocket,
  Satellite,
  Shield,
  Zap,
} from 'lucide-react';

export function MainNav() {
  const pathname = usePathname();

  const routes = [
    {
      href: '/',
      label: 'Dashboard',
      icon: <Home className="mr-2 h-4 w-4" />,
      active: pathname === '/',
    },
    {
      href: '/ccdm',
      label: 'CCDM',
      icon: <Shield className="mr-2 h-4 w-4" />,
      active: pathname === '/ccdm',
    },
    {
      href: '/satellites',
      label: 'Satellites',
      icon: <Satellite className="mr-2 h-4 w-4" />,
      active: pathname === '/satellites',
    },
    {
      href: '/conjunctions',
      label: 'Conjunctions',
      icon: <CircleAlert className="mr-2 h-4 w-4" />,
      active: pathname === '/conjunctions',
    },
    {
      href: '/trajectory',
      label: 'Trajectory',
      icon: <Zap className="mr-2 h-4 w-4" />,
      active: pathname === '/trajectory',
    },
    {
      href: '/maneuvers',
      label: 'Maneuvers',
      icon: <Rocket className="mr-2 h-4 w-4" />,
      active: pathname === '/maneuvers',
    },
    {
      href: '/catalog',
      label: 'Catalog',
      icon: <Boxes className="mr-2 h-4 w-4" />,
      active: pathname === '/catalog',
    },
    {
      href: '/analytics',
      label: 'Analytics',
      icon: <BarChart3 className="mr-2 h-4 w-4" />,
      active: pathname === '/analytics',
    },
  ];

  return (
    <nav className="flex items-center space-x-2">
      {routes.map((route) => (
        <Button
          key={route.href}
          variant={route.active ? 'default' : 'ghost'}
          className={cn(
            'justify-start',
            route.active ? 'bg-primary text-primary-foreground' : 'transparent'
          )}
          asChild
        >
          <Link href={route.href} className="flex items-center">
            {route.icon}
            {route.label}
          </Link>
        </Button>
      ))}
    </nav>
  );
} 