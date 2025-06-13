export default function NotFound() {
  return (
    <div className="flex h-screen w-full items-center justify-center bg-background">
      <div className="flex flex-col items-center space-y-6">
        <h1 className="text-4xl font-bold">404</h1>
        <p className="text-xl">Page not found</p>
        <a href="/" className="text-blue-500 hover:underline">Return to home</a>
      </div>
    </div>
  )
} 