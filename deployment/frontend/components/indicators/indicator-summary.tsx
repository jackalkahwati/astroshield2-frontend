import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const summaryData = [
  { title: "Total Indicators", value: "18" },
  { title: "ML Indicators", value: "12" },
  { title: "Rule Indicators", value: "5" },
  { title: "Threshold Indicators", value: "1" },
]

export function IndicatorSummary() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {summaryData.map((item) => (
        <Card key={item.title}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-white">{item.title}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{item.value}</div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

