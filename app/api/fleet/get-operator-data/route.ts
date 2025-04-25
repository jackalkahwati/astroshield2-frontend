import { NextResponse } from "next/server";

// Temporary stub for `/api/fleet/get-operator-data`
// Replace with real implementation once the backend endpoint is available.
export async function POST() {
  return NextResponse.json({
    success: true,
    payload: {
      operator_name: "Demo Operator",
      created_at: new Date().toISOString(),
    },
  });
}

export async function GET() {
  // Allow simple GET tests as well
  return POST();
} 