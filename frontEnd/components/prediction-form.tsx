"use client"
import { useRef } from "react"
import type React from "react"
import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Lightbulb, MessageSquare, Zap } from "lucide-react"
import Link from "next/link"

// Real data from the dataset
const cities = ["New York City", "Los Angeles", "Philadelphia", "San Francisco"]
const states = ["California", "England", "New York", "Texas"]
const countries = ["United States", "Australia", "France", "Mexico", "Germany"]
const products = ["Staples", "Cardinal Index Tab", "Eldon File Cart", "Rogers File Cart", "Ibico Index Tab"]
const priorities = ["Medium", "High", "Critical", "Low"]
const segments = ["Consumer", "Corporate", "Home Office"]
const shipModes = ["Standard Class", "Second Class", "First Class", "Same Day"]
const regions = ["Central", "South", "EMEA", "North", "Africa", "Oceania", "West", "Southeast Asia", "East"]
const categories = ["Office Supplies", "Technology", "Furniture"]
const subCategories = [
  "Binders",
  "Storage",
  "Art",
  "Paper",
  "Chairs",
  "Phones",
  "Furnishings",
  "Accessories",
  "Labels",
]
const quantities = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "14"]

const tips = [
  {
    title: "Need Sales Clarification?",
    description:
      "If the predicted sales value seems unclear or you want to understand the factors affecting it, our AI chatbot can provide detailed insights on how sales are calculated.",
    action: "Ask Chatbot",
  },
  {
    title: "Confused About Discount?",
    description:
      "Uncertain about the discount percentage? Chat with our AI to explore discount patterns, understand what influences discounts, and get optimization recommendations.",
    action: "Ask Chatbot",
  },
  {
    title: "Shipping Cost Breakdown?",
    description:
      "Want to understand the shipping cost better? Our chatbot can explain regional shipping factors, cost drivers, and ways to optimize shipping expenses.",
    action: "Ask Chatbot",
  },
]

export default function PredictionForm() {
  const [formData, setFormData] = useState({
    City: "",
    State: "",
    Country: "",
    ProductName: "",
    OrderPriority: "",
    Segment: "",
    ShipMode: "",
    Region: "",
    Category: "",
    SubCategory: "",
    Quantity: "",
    OrderDate: "",
  })

  const [predictions, setPredictions] = useState<{
    sales: number
    discount: number
    shippingCost: number
  } | null>(null)

  const [loading, setLoading] = useState(false)

    const resultRef = useRef<HTMLDivElement | null>(null)   // ← ضيفه هنا ✔️


  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  // ---------- FIXED HANDLE PREDICT ----------
  const handlePredict = async () => {
    setLoading(true)

    const payload = {
      City: formData.City,
      State: formData.State,
      Country: formData.Country,
      ProductName: formData.ProductName,
      OrderPriority: formData.OrderPriority,
      Segment: formData.Segment,
      ShipMode: formData.ShipMode,
      Region: formData.Region,
      Category: formData.Category,
      SubCategory: formData.SubCategory,
      Quantity: Number(formData.Quantity),
      OrderDate: formData.OrderDate,
    }

try {
  const res = await fetch("http://localhost:8001/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await res.json();

  console.log("SERVER RESPONSE:", data);

  if (!res.ok) {
    console.error("FastAPI returned error:", data);
    alert("Server Error: " + JSON.stringify(data));
    setLoading(false);
    return;
  }

  // --- Normalize keys to React format ---
  const sales = data.sales ?? null;
  const discount = data.discount ?? null;
  const shippingCost = data.shippingCost ?? data.shipping_cost ?? null;

  if (sales === null || discount === null || shippingCost === null) {
    alert("Prediction failed: Missing values in server response.");
    console.error("Invalid server response:", data);
    setLoading(false);
    return;
  }

  setPredictions({
    sales,
    discount,
    shippingCost,
  });

  // Scroll to results smoothly
setTimeout(() => {
  resultRef.current?.scrollIntoView({ behavior: "smooth" })
}, 200)


} catch (err) {
  console.error("Error sending data to FastAPI:", err);
  alert("Cannot reach backend server. Make sure FastAPI is running.");
} finally {
  setLoading(false);
}
  }

  const randomTip = tips[Math.floor(Math.random() * tips.length)]

  return (
    <div className="space-y-8">
      {/* Input Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Location Section */}
        <div className="lg:col-span-3">
          <h3 className="text-lg font-semibold mb-4 gradient-text">🏠︎ Location Details</h3>
        </div>

{/* CITY */}
<Card className="p-5 card-glow fade-in-scale">
  <Label htmlFor="City" className="text-sm font-medium block mb-2">
    City
  </Label>
  <input
    type="text"
    id="City"
    name="City"
    placeholder="Enter City"
    value={formData.City}
    onChange={handleInputChange}
    className="w-full px-3 py-2 bg-slate-800 text-foreground border border-purple-500/30 rounded-lg"
  />
</Card>

{/* STATE */}
<Card className="p-5 card-glow fade-in-scale">
  <Label htmlFor="State" className="text-sm font-medium block mb-2">
    State
  </Label>
  <input
    type="text"
    id="State"
    name="State"
    placeholder="Enter State"
    value={formData.State}
    onChange={handleInputChange}
    className="w-full px-3 py-2 bg-slate-800 text-foreground border border-purple-500/30 rounded-lg"
  />
</Card>

{/* COUNTRY */}
<Card className="p-5 card-glow fade-in-scale">
  <Label htmlFor="Country" className="text-sm font-medium block mb-2">
    Country
  </Label>
  <input
    type="text"
    id="Country"
    name="Country"
    placeholder="Enter Country"
    value={formData.Country}
    onChange={handleInputChange}
    className="w-full px-3 py-2 bg-slate-800 text-foreground border border-purple-500/30 rounded-lg"
  />
</Card>
        {/* Product Details Section */}
        <div className="lg:col-span-3">
          <h3 className="text-lg font-semibold mb-4 gradient-text">🛒 Product Details</h3>
        </div>
    

{/* PRODUCT NAME */}
<Card className="p-5 card-glow fade-in-scale">
  <Label htmlFor="ProductName" className="text-sm font-medium block mb-2">
    Product Name
  </Label>
  <input
    type="text"
    id="ProductName"
    name="ProductName"
    placeholder="Enter Product Name"
    value={formData.ProductName}
    onChange={handleInputChange}
    className="w-full px-3 py-2 bg-slate-800 text-foreground border border-purple-500/30 rounded-lg"
  />
</Card>

{/* CATEGORY */}
<Card className="p-5 card-glow fade-in-scale">
  <Label htmlFor="Category" className="text-sm font-medium block mb-2">
    Category
  </Label>
  <select
    id="Category"
    name="Category"
    value={formData.Category}
    onChange={handleInputChange}
    className="w-full px-3 py-2 bg-slate-800 text-foreground border border-purple-500/30 rounded-lg"
  >
    <option value="">Select Category</option>
    <option value="Office Supplies">Office Supplies</option>
    <option value="Technology">Technology</option>
    <option value="Furniture">Furniture</option>
  </select>
</Card>

{/* SUB-CATEGORY */}
<Card className="p-5 card-glow fade-in-scale">
  <Label htmlFor="SubCategory" className="text-sm font-medium block mb-2">
    Sub-Category
  </Label>
  <select
    id="SubCategory"
    name="SubCategory"
    value={formData.SubCategory}
    onChange={handleInputChange}
    className="w-full px-3 py-2 bg-slate-800 text-foreground border border-purple-500/30 rounded-lg"
  >
    <option value="">Select Sub-Category</option>
    <option value="Binders">Binders</option>
    <option value="Storage">Storage</option>
    <option value="Art">Art</option>
    <option value="Paper">Paper</option>
    <option value="Chairs">Chairs</option>
    <option value="Phones">Phones</option>
    <option value="Furnishings">Furnishings</option>
    <option value="Accessories">Accessories</option>
    <option value="Labels">Labels</option>
    <option value="Envelopes">Envelopes</option>
    <option value="Supplies">Supplies</option>
    <option value="Fasteners">Fasteners</option>
    <option value="Bookcases">Bookcases</option>
    <option value="Copiers">Copiers</option>
    <option value="Appliances">Appliances</option>
    <option value="Machines">Machines</option>
    <option value="Tables">Tables</option>
  </select>
</Card>

{/* ORDER PRIORITY */}
<Card className="p-5 card-glow fade-in-scale">
  <Label htmlFor="OrderPriority" className="text-sm font-medium block mb-2">
    Order Priority
  </Label>
  <select
    id="OrderPriority"
    name="OrderPriority"
    value={formData.OrderPriority}
    onChange={handleInputChange}
    className="w-full px-3 py-2 bg-slate-800 text-foreground border border-purple-500/30 rounded-lg"
  >
    <option value="">Select Priority</option>
    <option value="Medium">Medium</option>
    <option value="High">High</option>
    <option value="Critical">Critical</option>
    <option value="Low">Low</option>
  </select>
</Card>

{/* SEGMENT */}
<Card className="p-5 card-glow fade-in-scale">
  <Label htmlFor="Segment" className="text-sm font-medium block mb-2">
    Segment
  </Label>
  <select
    id="Segment"
    name="Segment"
    value={formData.Segment}
    onChange={handleInputChange}
    className="w-full px-3 py-2 bg-slate-800 text-foreground border border-purple-500/30 rounded-lg"
  >
    <option value="">Select Segment</option>
    <option value="Consumer">Consumer</option>
    <option value="Corporate">Corporate</option>
    <option value="Home Office">Home Office</option>
  </select>
</Card>

{/* SHIP MODE */}
<Card className="p-5 card-glow fade-in-scale">
  <Label htmlFor="ShipMode" className="text-sm font-medium block mb-2">
    Ship Mode
  </Label>
  <select
    id="ShipMode"
    name="ShipMode"
    value={formData.ShipMode}
    onChange={handleInputChange}
    className="w-full px-3 py-2 bg-slate-800 text-foreground border border-purple-500/30 rounded-lg"
  >
    <option value="">Select Ship Mode</option>
    <option value="Standard Class">Standard Class</option>
    <option value="Second Class">Second Class</option>
    <option value="First Class">First Class</option>
    <option value="Same Day">Same Day</option>
  </select>
</Card>

        {/* Order Details Section */}
        <div className="lg:col-span-3">
          <h3 className="text-lg font-semibold mb-4 gradient-text">🚚 Order Details</h3>
        </div>

{/* REGION */}
<Card className="p-5 card-glow fade-in-scale">
  <Label htmlFor="Region" className="text-sm font-medium block mb-2">
    Region
  </Label>
  <input
    type="text"
    id="Region"
    name="Region"
    placeholder="Enter Region"
    value={formData.Region}
    onChange={handleInputChange}
    className="w-full px-3 py-2 bg-slate-800 text-foreground border border-purple-500/30 rounded-lg"
  />
</Card>

{/* QUANTITY */}
<Card className="p-5 card-glow fade-in-scale">
  <Label htmlFor="Quantity" className="text-sm font-medium block mb-2">
    Quantity
  </Label>
  <Input
    id="Quantity"
    name="Quantity"
    type="number"
    min="1"
    max="100"
    placeholder="Enter quantity (1-100)"
    value={formData.Quantity}
    onChange={handleInputChange}
    className="bg-slate-800 text-foreground border-blue-500/30 focus:ring-blue-500 focus:border-blue-500"
  />
</Card>


        <Card className="p-5 card-glow fade-in-scale">
          <Label htmlFor="OrderDate" className="text-sm font-medium block mb-2">
            Order Date
          </Label>
          <Input
            id="OrderDate"
            name="OrderDate"
            type="date"
            value={formData.OrderDate}
            onChange={handleInputChange}
            className="bg-slate-800 text-foreground border-purple-500/30 focus:ring-purple-500"
          />
        </Card>
      </div>

      {/* Predict Button */}
      <div className="flex justify-center">
        <Button
          onClick={handlePredict}
          disabled={loading}
          className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-semibold py-3 px-8 rounded-lg shadow-lg shadow-blue-500/50 hover:shadow-blue-500/70 transition-all duration-300 flex items-center gap-2"
        >
          <Zap size={20} />
          {loading ? "Predicting..." : "Generate Prediction"}
        </Button>
      </div>


      {/* Predictions Results */}
<div ref={resultRef}>
  {predictions && (
    <div className="space-y-6 slide-up">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="p-6 card-glow bg-gradient-to-br from-blue-600/10 to-cyan-600/10 border-blue-500/50">
          <p className="text-sm text-muted-foreground mb-2">Predicted Sales</p>
          <p className="text-4xl font-bold text-blue-400">${predictions.sales.toFixed(2)}</p>
        </Card>

        <Card className="p-6 card-glow bg-gradient-to-br from-cyan-600/10 to-teal-600/10 border-cyan-500/50">
          <p className="text-sm text-muted-foreground mb-2">Estimated Discount</p>
          <p className="text-4xl font-bold text-cyan-400">{(predictions.discount * 100).toFixed(1)}%</p>
        </Card>

        <Card className="p-6 card-glow bg-gradient-to-br from-purple-600/10 to-blue-600/10 border-purple-500/50">
          <p className="text-sm text-muted-foreground mb-2">Shipping Cost</p>
          <p className="text-4xl font-bold text-purple-400">${predictions.shippingCost.toFixed(2)}</p>
        </Card>
      </div>

      <Card className="p-6 card-glow bg-gradient-to-br from-slate-800 to-slate-900 border-cyan-500/30">
        <div className="flex items-start gap-4">
          <div className="bg-gradient-to-br from-cyan-500 to-blue-500 p-3 rounded-lg flex-shrink-0">
            <Lightbulb size={24} className="text-white" />
          </div>
          <div className="flex-1">
            <h4 className="font-semibold text-cyan-300 mb-1">💡 {randomTip.title}</h4>
            <p className="text-sm text-muted-foreground mb-3">{randomTip.description}</p>
            <p className="text-xs text-muted-foreground mb-3">
              If any of these statistics seem unclear, our AI chatbot is here to help explain the details.
            </p>
            <Link href="/chatbot">
              <Button className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white text-sm flex items-center gap-2">
                <MessageSquare size={16} />
                {randomTip.action}
              </Button>
            </Link>
          </div>
        </div>
      </Card>
    </div>
  )}
</div>
    </div>
  )
} 