import LayoutWrapper from "@/components/layout-wrapper"
import PredictionForm from "@/components/prediction-form"

export const metadata = {
  title: "Sales Prediction - SalesForce AI",
  description: "Predict sales, discount, and shipping costs using advanced AI",
}

export default function Home() {
  return (
    <LayoutWrapper>
      <div className="space-y-8">
        <div>
          <h1 className="text-5xl font-bold mb-2 gradient-text">Sales Prediction</h1>
          <p className="text-lg text-muted-foreground">Enter your data and get instant AI-powered sales predictions</p>
        </div>
        <PredictionForm />
      </div>
    </LayoutWrapper>
  )
}
