import { NextResponse } from "next/server";

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
  return POST();
} 