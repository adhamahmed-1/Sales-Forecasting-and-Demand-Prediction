"use client"

import React, { useState } from "react"
import Papa from "papaparse"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, TrendingUp, DollarSign, Percent } from "lucide-react"
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"

const COLORS = ["#ef4444", "#fbbf24", "#10b981"]

const KPI_COLORS = {
  sales: {
    gradient: "from-blue-600/10 to-cyan-600/10",
    border: "border-blue-500/50",
    icon: "bg-gradient-to-br from-blue-500 to-cyan-500",
  },
  discount: {
    gradient: "from-emerald-600/10 to-teal-600/10",
    border: "border-emerald-500/50",
    icon: "bg-gradient-to-br from-emerald-500 to-teal-500",
  },
  orders: {
    gradient: "from-purple-600/10 to-pink-600/10",
    border: "border-purple-500/50",
    icon: "bg-gradient-to-br from-purple-500 to-pink-500",
  },
}

export default function Dashboard() {
  const [fileName, setFileName] = useState("")
  const [parsedData, setParsedData] = useState<any[]>([])
  const [error, setError] = useState("")

  const REQUIRED_COLUMNS = ["Sales", "Category", "Segment", "Discount"]
  const OPTIONAL_ORDER_COLUMNS = ["Orders", "Quantity"]

  const validateColumns = (columns: string[]) => {
    for (let col of REQUIRED_COLUMNS) {
      if (!columns.includes(col)) return false
    }

    const hasOrders = OPTIONAL_ORDER_COLUMNS.some((c) => columns.includes(c))
    if (!hasOrders) return false

    return true
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setFileName(file.name)
    setError("")

    interface ParsedRow {
      Sales: number
      Category: string
      Segment: string
      Discount: number
      Orders: number
      [key: string]: string | number
    }

    interface ParseResults {
      data: ParsedRow[]
    }

    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results: ParseResults) => {
        const rows = results.data as ParsedRow[]
        const columns = Object.keys(rows[0] || {})

        if (!validateColumns(columns)) {
          setParsedData([])
          setError(
            "❌ CSV must include at least: Sales, Category, Segment, Discount, and Orders or Quantity."
          )
          return
        }

        const ordersCol = columns.includes("Orders") ? "Orders" : "Quantity"

        const cleaned: ParsedRow[] = rows.map((r) => ({
          ...r,
          Sales: Number(r.Sales),
          Discount: Number(r.Discount),
          Orders: Number(r[ordersCol]),
        }))

        setParsedData(cleaned)
      },
    })
  }

  const totalSales = parsedData.reduce((a, b) => a + (b.Sales || 0), 0)
  const avgDiscount =
    parsedData.length > 0
      ? parsedData.reduce((a, b) => a + (b.Discount || 0), 0) / parsedData.length
      : 0
  const totalOrders = parsedData.reduce((a, b) => a + (b.Orders || 0), 0)

  const salesByMonth = parsedData.slice(0, 6).map((row, i) => ({
    month: `M${i + 1}`,
    sales: row.Sales,
    orders: row.Orders,
  }))

  const categoryTotals: { name: string; value: number }[] = Object.values(
    parsedData.reduce((acc: any, row) => {
      if (!acc[row.Category]) acc[row.Category] = { name: row.Category, value: 0 }
      acc[row.Category].value += row.Sales
      return acc
    }, {})
  )

  const segmentTotals = Object.values(
    parsedData.reduce((acc: any, row) => {
      if (!acc[row.Segment])
        acc[row.Segment] = { segment: row.Segment, sales: 0, discount: 0 }
      acc[row.Segment].sales += row.Sales
      acc[row.Segment].discount += row.Discount
      return acc
    }, {})
  ).map((s: any) => ({
    segment: s.segment,
    sales: s.sales,
    discount: s.discount / parsedData.length,
  }))

  return (
    <div className="space-y-6">
      {/* Upload Section */}
      <Card className="p-6 card-glow bg-gradient-to-br from-slate-800 to-slate-900 border-blue-500/30 fade-in-scale">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold mb-2 text-foreground">Data Upload</h2>
            <p className="text-sm text-muted-foreground">
              {fileName ? `Loaded: ${fileName}` : "Upload a CSV or Excel file to get started"}
            </p>
          </div>
          <label className="cursor-pointer relative inline-block">
  <Button
    type="button"
    className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-cyan-600 
               hover:from-blue-700 hover:to-cyan-700 text-white shadow-lg shadow-blue-500/50"
  >
    <Upload size={18} />
    Choose File
  </Button>

  <input
    type="file"
    accept=".csv"
    onChange={handleFileUpload}
    className="absolute inset-0 opacity-0 cursor-pointer"
    style={{ width: "100%", height: "100%" }}
  />
</label>


        </div>

        {error && <p className="text-red-400 mt-3">{error}</p>}
      </Card>

      {/* If no data loaded → Stop */}
      {!parsedData.length ? null : (
        <>
          {/* KPIs */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card
              className={`p-6 card-glow bg-gradient-to-br ${KPI_COLORS.sales.gradient} ${KPI_COLORS.sales.border} fade-in-scale`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm text-muted-foreground mb-2">Total Sales</p>
                  <p className="text-3xl font-bold text-blue-400">${totalSales.toFixed(2)}</p>
                </div>
                <div className={`p-3 ${KPI_COLORS.sales.icon} rounded-lg`}>
                  <DollarSign className="text-white" size={24} />
                </div>
              </div>
            </Card>

            <Card
              className={`p-6 card-glow bg-gradient-to-br ${KPI_COLORS.discount.gradient} ${KPI_COLORS.discount.border} fade-in-scale`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm text-muted-foreground mb-2">Avg Discount</p>
                  <p className="text-3xl font-bold text-emerald-400">
                    {(avgDiscount * 100).toFixed(1)}%
                  </p>
                </div>
                <div className={`p-3 ${KPI_COLORS.discount.icon} rounded-lg`}>
                  <Percent className="text-white" size={24} />
                </div>
              </div>
            </Card>

            <Card
              className={`p-6 card-glow bg-gradient-to-br ${KPI_COLORS.orders.gradient} ${KPI_COLORS.orders.border} fade-in-scale`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm text-muted-foreground mb-2">Total Orders</p>
                  <p className="text-3xl font-bold text-purple-400">{totalOrders}</p>
                </div>
                <div className={`p-3 ${KPI_COLORS.orders.icon} rounded-lg`}>
                  <TrendingUp className="text-white" size={24} />
                </div>
              </div>
            </Card>
          </div>

          {/* Sales Trend */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="p-6 card-glow bg-gradient-to-br from-slate-800 to-slate-900 border-blue-500/30 fade-in-scale">
              <h3 className="text-lg font-semibold mb-4 gradient-text">Sales Trend</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={salesByMonth}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 32.6% 17.5%)" />
                  <XAxis dataKey="month" stroke="hsl(215 20.2% 65.1%)" />
                  <YAxis stroke="hsl(215 20.2% 65.1%)" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(222 14% 13%)",
                      border: "1px solid rgba(100, 200, 255, 0.3)",
                      borderRadius: "8px",
                    }}
                    labelStyle={{ color: "hsl(210 40% 98%)" }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="sales" stroke="#10b981" strokeWidth={3} dot={{ r: 5 }} />
                  <Line type="monotone" dataKey="orders" stroke="#ef4444" strokeWidth={3} dot={{ r: 5 }} />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            {/* Category Distribution */}
            <Card className="p-6 card-glow bg-gradient-to-br from-slate-800 to-slate-900 border-emerald-500/30 fade-in-scale">
              <h3 className="text-lg font-semibold mb-4 gradient-text">Sales by Category</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={categoryTotals}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`}
                    outerRadius={100}
                    dataKey="value"
                  >
                    {categoryTotals.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(222 14% 13%)",
                      border: "1px solid rgba(16, 185, 129, 0.3)",
                      borderRadius: "8px",
                    }}
                    labelStyle={{ color: "hsl(210 40% 98%)" }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Segment Performance */}
          <Card className="p-6 card-glow bg-gradient-to-br from-slate-800 to-slate-900 border-purple-500/30 lg:col-span-2 fade-in-scale">
            <h3 className="text-lg font-semibold mb-4 gradient-text">Segment Performance</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={segmentTotals}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 32.6% 17.5%)" />
                <XAxis dataKey="segment" stroke="hsl(215 20.2% 65.1%)" />
                <YAxis stroke="hsl(215 20.2% 65.1%)" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(222 14% 13%)",
                    border: "1px solid rgba(168, 85, 247, 0.3)",
                    borderRadius: "8px",
                  }}
                  labelStyle={{ color: "hsl(210 40% 98%)" }}
                />
                <Legend />
                <Bar dataKey="sales" fill="#a855f7" name="Sales" />
                <Bar dataKey="discount" fill="#10b981" name="Discount" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Data Table */}
          <Card className="p-6 card-glow bg-gradient-to-br from-slate-800 to-slate-900 border-purple-500/30 fade-in-scale">
            <h3 className="text-lg font-semibold mb-4 gradient-text">Data Preview</h3>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-purple-500/30">
                    {Object.keys(parsedData[0]).map((col) => (
                      <th key={col} className="text-left py-3 px-4 text-muted-foreground font-medium">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>

                <tbody>
                  {parsedData.slice(0, 20).map((row, i) => (
                    <tr
                      key={i}
                      className="border-b border-purple-500/20 hover:bg-purple-600/10 transition-colors"
                    >
                      {Object.values(row).map((cell: any, j) => (
                        <td key={j} className="py-3 px-4 text-foreground">
                          {String(cell)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
