#!/usr/bin/env python3
"""Industry taxonomy with NATURAL-LANGUAGE triggers — teaches the model to INFER
the right industry/objectives from how a real person describes their idea
("I want to sell lemonade" → Food & Beverage), instead of being handed the tag.

Each industry maps to: natural business-idea phrases (the user's words), the
platforms/objectives that idea typically implies, and which app archetypes fit.
This is both the inference signal AND a large entropy source for the generator.
"""

INDUSTRIES = {
    "Food & Beverage": {
        "ideas": ["sell lemonade at events", "run a coffee cart", "open a bakery",
                  "start a taco food truck", "deliver homemade meals",
                  "run a juice and smoothie bar", "a meal-prep subscription",
                  "a craft brewery taproom", "a farmers-market produce stand",
                  "a pizza delivery joint", "a bubble tea shop",
                  "cater weddings and events", "a ghost kitchen for delivery",
                  "a coffee roastery with a subscription", "a dessert pop-up"],
        "platforms": [["iOS", "Android"], ["iOS", "Android", "Web"]],
        "objectives": ["Online ordering", "Menu management", "Delivery",
                       "Scheduled pickup", "Loyalty rewards", "Payments",
                       "Reservations"],
        "apps": ["ordering", "marketplace", "booking"],
    },
    "Health & Fitness": {
        "ideas": ["help people track workouts", "a personal-training booking app",
                  "a running and cycling tracker", "a yoga class scheduler",
                  "a nutrition and calorie log", "a gym membership app",
                  "a meditation and sleep app", "a physical-therapy exercise app",
                  "a step-and-activity challenge app", "a habit and wellness "
                  "tracker", "a CrossFit box management app"],
        "platforms": [["iOS", "Android"], ["iOS", "Android", "Web"]],
        "objectives": ["Workout logging", "Training plans", "Progress charts",
                       "Class booking", "Reminders", "Social challenges"],
        "apps": ["tracker", "booking", "social"],
    },
    "Retail": {
        "ideas": ["run a clothing boutique", "a sneaker resale shop",
                  "a hardware store with inventory", "a bookstore",
                  "a pet supply shop", "a thrift store", "a comic book store",
                  "a plant nursery", "a toy store", "a jewelry shop",
                  "track stock across two store locations"],
        "platforms": [["iOS", "Android"], ["Web"], ["iOS", "Android", "Web"]],
        "objectives": ["Inventory", "Point of sale", "Barcode scanning",
                       "Low-stock alerts", "Reorder", "Reports", "Loyalty"],
        "apps": ["inventory", "marketplace", "ordering"],
    },
    "Services": {
        "ideas": ["book appointments for a salon", "a barbershop scheduler",
                  "a dog grooming booking app", "a house-cleaning service",
                  "a handyman dispatch app", "a tattoo studio scheduler",
                  "a massage therapy booking app", "a tutoring scheduler",
                  "a car-detailing booking app", "a lawn-care scheduling app"],
        "platforms": [["iOS", "Android"], ["iOS", "Android", "Web"], ["Web"]],
        "objectives": ["Appointment booking", "Calendar sync", "Reminders",
                       "Staff scheduling", "Deposits", "Reviews"],
        "apps": ["booking", "crm", "marketplace"],
    },
    "Business / SaaS": {
        "ideas": ["a CRM for a small sales team", "a project management tool",
                  "an invoicing app for freelancers", "a help-desk for support",
                  "an HR onboarding tool", "a contract management app",
                  "a team wiki and docs tool", "an expense-tracking app",
                  "a recruiting pipeline tool", "an OKR tracking app"],
        "platforms": [["Web"], ["Web", "Cloud/Server"]],
        "objectives": ["Contact management", "Pipeline", "Reporting", "Tasks",
                       "Reminders", "Integrations", "Roles & permissions"],
        "apps": ["crm", "dashboard", "helpdesk"],
    },
    "Telephony / IVR": {
        "ideas": ["an AI phone system that answers calls",
                  "an automated appointment-reminder caller",
                  "a call-routing menu for a clinic",
                  "an outbound survey calling system",
                  "a voicemail-to-text service", "a support call deflection bot",
                  "a phone-based order line for a restaurant"],
        "platforms": [["Cloud/Server"], ["Web", "Cloud/Server"]],
        "objectives": ["Inbound routing", "Menu flows", "Voicemail",
                       "Outbound calls", "Speech recognition", "Call analytics"],
        "apps": ["ivr", "dashboard"],
    },
    "Education": {
        "ideas": ["an online course platform", "a language-learning app",
                  "a quiz and flashcard app", "a coding bootcamp LMS",
                  "a tutoring marketplace", "a school assignment tracker",
                  "a music-lesson scheduling app", "a kids' reading app",
                  "a certification exam prep app"],
        "platforms": [["iOS", "Android", "Web"], ["Web"]],
        "objectives": ["Courses", "Lessons", "Quizzes", "Progress tracking",
                       "Certificates", "Enrollment", "Discussions"],
        "apps": ["lms", "marketplace", "social"],
    },
    "Fintech": {
        "ideas": ["a personal budgeting app", "a peer-to-peer payment app",
                  "an expense-splitting app for roommates", "an invoicing and "
                  "payments tool", "a savings-goal app", "a crypto portfolio "
                  "tracker", "a small-business banking dashboard",
                  "a subscription-tracking app", "a tip-pooling app for staff"],
        "platforms": [["iOS", "Android"], ["iOS", "Android", "Web"], ["Web"]],
        "objectives": ["Accounts", "Transfers", "Budgeting", "Statements",
                       "Alerts", "Bill pay", "Categorization"],
        "apps": ["fintech", "dashboard", "tracker"],
    },
    "Marketplace": {
        "ideas": ["a marketplace for handmade crafts", "a used-furniture "
                  "marketplace", "a local services marketplace",
                  "a freelance gig marketplace", "a rental marketplace for tools",
                  "a farmers-to-restaurants marketplace", "a vintage clothing "
                  "marketplace", "a car parts marketplace"],
        "platforms": [["iOS", "Android", "Web"], ["Web"]],
        "objectives": ["Listings", "Search & filters", "Messaging", "Checkout",
                       "Reviews", "Seller payouts", "Disputes"],
        "apps": ["marketplace", "ordering", "social"],
    },
    "Logistics": {
        "ideas": ["a delivery dispatch app", "a courier tracking app",
                  "a fleet management tool", "a last-mile delivery app",
                  "a moving-company scheduling app", "a freight booking platform",
                  "a field-service routing app"],
        "platforms": [["iOS", "Android"], ["iOS", "Android", "Web",
                                            "Cloud/Server"]],
        "objectives": ["Dispatch", "Route optimization", "Live tracking",
                       "Proof of delivery", "ETAs", "Notifications"],
        "apps": ["logistics", "dashboard"],
    },
    "Social": {
        "ideas": ["a community app for hikers", "a social app for book clubs",
                  "a niche forum for gardeners", "a photo-sharing app for pets",
                  "a local events and meetup app", "a hobby-sharing community",
                  "a neighborhood bulletin-board app", "a fan community app"],
        "platforms": [["iOS", "Android"], ["iOS", "Android", "Web"]],
        "objectives": ["Posting", "Feed", "Following", "Comments",
                       "Notifications", "Direct messages", "Groups",
                       "Moderation"],
        "apps": ["social", "marketplace"],
    },
    "Media / CMS": {
        "ideas": ["a blog and content site", "a news publishing platform",
                  "a podcast hosting site", "a recipe website with a CMS",
                  "a documentation site", "a portfolio site builder",
                  "a digital magazine"],
        "platforms": [["Web"], ["Web", "Cloud/Server"]],
        "objectives": ["Content authoring", "Publishing", "Media library",
                       "SEO", "Scheduling", "Versioning", "Roles"],
        "apps": ["cms", "dashboard"],
    },
    "Real Estate": {
        "ideas": ["a property-listing app", "a rental-management tool for "
                  "landlords", "a home-search app", "a vacation-rental booking "
                  "site", "a tenant-maintenance request app", "a real-estate "
                  "CRM for agents"],
        "platforms": [["iOS", "Android", "Web"], ["Web"]],
        "objectives": ["Listings", "Search & filters", "Booking", "Messaging",
                       "Maintenance requests", "Payments", "Reports"],
        "apps": ["marketplace", "booking", "crm"],
    },
    "Travel & Hospitality": {
        "ideas": ["a hotel booking app", "a tour-booking platform",
                  "a campground reservation app", "a restaurant reservation app",
                  "a city-guide travel app", "a flight-deal tracker",
                  "a vacation itinerary planner"],
        "platforms": [["iOS", "Android"], ["iOS", "Android", "Web"]],
        "objectives": ["Booking", "Search", "Itineraries", "Reviews",
                       "Payments", "Reminders", "Maps"],
        "apps": ["booking", "marketplace", "tracker"],
    },
    "Analytics": {
        "ideas": ["a sales analytics dashboard", "a marketing metrics dashboard",
                  "a website-traffic dashboard", "an IoT sensor dashboard",
                  "a finance reporting dashboard", "an app-usage analytics tool"],
        "platforms": [["Web"], ["Web", "Cloud/Server"]],
        "objectives": ["Reporting", "KPIs", "Charts", "Filters", "Exports",
                       "Alerts", "Scheduled reports"],
        "apps": ["dashboard", "tracker"],
    },
    "Support": {
        "ideas": ["a customer support help desk", "a ticketing system for IT",
                  "a knowledge base and FAQ site", "a live-chat support tool",
                  "a bug-report tracker", "a returns and RMA portal"],
        "platforms": [["Web"], ["Web", "Cloud/Server"]],
        "objectives": ["Ticketing", "Queues", "SLAs", "Knowledge base",
                       "Routing", "CSAT surveys", "Reporting"],
        "apps": ["helpdesk", "dashboard"],
    },
}

# Casual ways a real user opens with their idea. We keep two registers so the
# grammar stays natural: VERB openers for activity phrases ("sell lemonade") and
# NOUN openers for thing phrases ("a bakery", "an OKR tracking app").
VERB_OPENERS = [
    "I want to {idea}.",
    "Basically I {idea} and need an app for it.",
    "I {idea}, and I want to turn that into an app.",
    "We help people {idea} — need software for it.",
    "I run a business where I {idea}; I need an app.",
    "thinking of making an app to {idea}, not sure where to start",
    "Could we make an app that lets me {idea}?",
    "I'd love an app to {idea} — what do you think?",
    "My plan is to {idea}.",
]
NOUN_OPENERS = [
    "I want to build {idea}.",
    "My idea is {idea}.",
    "I'm thinking about {idea}.",
    "We're building {idea}.",
    "Looking to launch {idea}.",
    "Help me plan {idea}.",
    "{idea} — that's what I want to build.",
    "so i want to make {idea}, where do we start?",
    "Could you help me build {idea}?",
]


def opener_for(idea):
    """Pick the right opener register for an idea phrase (noun vs verb)."""
    first = idea.strip().split()[0].lower()
    return NOUN_OPENERS if first in ("a", "an", "the") else VERB_OPENERS

# Assistant one-line reflections after inferring the industry (paraphrased).
INFER_REFLECTIONS = [
    "Sounds like a {ind} product — tagging that to start.",
    "Got it, that's {ind}. Let me record the basics.",
    "That reads as {ind}; I'll tag it and fill in the rest.",
    "Nice — {ind}. Capturing what that implies.",
    "I'll file this under {ind} and build from there.",
]
