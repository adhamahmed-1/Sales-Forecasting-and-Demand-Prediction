"use client"

import { Card } from "@/components/ui/card"
import { Linkedin } from "lucide-react"

interface TeamMember {
  id: string
  name: string
  role: string
  description: string
  image: string
  linkedin?: string
}

const teamMembers: TeamMember[] = [
  {
    id: "1",
    name: "Ahmed Lotfy",
    role: "Machine Learning Engineer",
    description: "Specialized in developing and optimizing ML models, designing scalable data pipelines and production-grade AI solutions.",
    image: "/lootf.jpg",
    linkedin: "https://www.linkedin.com/in/ahmed-lotfy-6ab263329/",
  },
  {
    id: "2",
    name: "Adham Ahmed",
    role: "Full Stack Developer",
    description: "Skilled in developing end-to-end applications, designing scalable APIs and DevOps tools to deliver efficient, production-ready systems.",
    image: "/aaadham.jpg",
    linkedin: "https://www.linkedin.com/in/adham-ahmed-8a389b371/",
  },
  {
    id: "3",
    name: "Moaz Shaban",
    role: "Generative AI Engineer",
    description: "Builds real-world applications powered by generative AI, LLMs, and intelligent automation.",
    image: "/zzz.jpg",
    linkedin: "https://www.linkedin.com/in/moaz-shaban-a3b674283/",
  },
  {
    id: "4",
    name: "Arwa Ramadan",
    role: "Data Analyst",
    description: "Transforms data into actionable insights for business growth.",
    image: "/noo.png",
    linkedin: "https://www.linkedin.com/in/arwa-ramadan/",
  },
  {
    id: "5",
    name: "Amani Habib",
    role: "Data Engineer",
    description: "Builds and maintains data pipelines and infrastructure to ensure efficient and reliable data flow.",
    image: "/amaani.jpg",
    linkedin: "https://www.linkedin.com/in/amani-habeeb/",
  }
]

export default function TeamSection() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {teamMembers.map((member, index) => (
        <Card
          key={member.id}
          className="overflow-hidden card-glow border-purple-500/30 fade-in-scale hover:border-purple-500/60 transition-all duration-300"
          style={{ animationDelay: `${index * 50}ms` }}
        >
          {/* Member Image */}
          <div className="relative h-64 overflow-hidden bg-gradient-to-br from-purple-600/20 to-blue-600/20">
            <img src={member.image} alt={member.name} className="w-full h-full object-cover" />
            <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/20 to-transparent"></div>
          </div>

          {/* Member Info */}
          <div className="p-6 bg-gradient-to-br from-slate-800 to-slate-900">
            <h3 className="text-lg font-bold gradient-text">{member.name}</h3>
            <p className="text-sm font-semibold mb-2 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
              {member.role}
            </p>
            <p className="text-sm text-muted-foreground mb-4 leading-relaxed">{member.description}</p>

            {/* Social Links */}
            <div className="flex gap-3 pt-4 border-t border-purple-500/20">
              <a
                href={member.linkedin}
                className="p-2 rounded-lg bg-gradient-to-br from-purple-600/20 to-blue-600/20 hover:from-purple-600/40 hover:to-blue-600/40 text-muted-foreground hover:text-blue-400 transition-all duration-300 border border-purple-500/20 hover:border-blue-500/50"
                aria-label={`LinkedIn profile of ${member.name}`}
              >
                <Linkedin size={18} />
              </a>
            </div>
          </div>
        </Card>
      ))}
    </div>
  )
}
