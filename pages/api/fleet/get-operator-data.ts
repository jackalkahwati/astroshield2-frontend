import type { NextApiRequest, NextApiResponse } from 'next'

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  res.status(200).json({
    success: true,
    payload: {
      operator_name: "Demo Operator",
      created_at: new Date().toISOString(),
    },
  })
} 