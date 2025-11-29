"use client"

import LayoutWrapper from "@/components/layout-wrapper"
import TeamSection from "@/components/team-section"

export default function TeamPage() {
  return (
    <LayoutWrapper>
      <div className="space-y-12">
        <div className="text-center max-w-3xl mx-auto">
          <h1 className="text-5xl font-bold mb-4 gradient-text">Our Team</h1>
          <p className="text-lg text-muted-foreground">
            Meet the talented professionals behind the SalesForce AI platform. Together, we're revolutionizing sales
            prediction with machine learning.
          </p>
        </div>
        <TeamSection />
      </div>
    </LayoutWrapper>
  )
}
