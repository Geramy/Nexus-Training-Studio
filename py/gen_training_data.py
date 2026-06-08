#!/usr/bin/env python3
"""Procedural generator for the Nexus combined model: tens of thousands of
high-quality, FORMAT-CORRECT tool-calling conversations for (1) Setup interview,
(2) Discovery / user-story building, and (3) Task generation.

Diversity is the #1 quality lever (research: +7.4% BFCL from request-phrasing +
argument-value diversity), so scenarios are sampled combinatorially from large
pools (domain nouns × app types × platform sets × stacks × phrasings) and each
yields several randomized conversation variants. Every example carries the REAL
tool schemas (mlx_lm "tools" format) and the app's exact message shapes, so
train == serve. Output is appended (deduped) to workspace/data/dataset.jsonl.

Usage:
  gen_training_data.py [--target 10000] [--kinds setup,discovery,tasks] [--seed 7]
"""
import argparse
import json
import random
import sys

from data_common import append_conversations
from tool_schemas import tools_for
from taxonomy import INDUSTRIES, INFER_REFLECTIONS, opener_for

# ───────────────────────── message + tool-call helpers ─────────────────────


class Ids:
    def __init__(self):
        self.n = 0

    def next(self):
        self.n += 1
        return f"c{self.n}"


def sys_msg(t):
    return {"role": "system", "content": t}


def user_msg(t):
    return {"role": "user", "content": t}


def asst_text(t):
    return {"role": "assistant", "content": t}


def tool_call(ids, name, args):
    cid = ids.next()
    return cid, {"id": cid, "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}


def asst_calls(calls, content=None):
    return {"role": "assistant", "content": content if content else None,
            "tool_calls": calls}


def tool_result(cid, obj, name=None):
    m = {"role": "tool", "tool_call_id": cid,
         "content": obj if isinstance(obj, str) else json.dumps(obj)}
    if name:
        m["name"] = name
    return m


# ───────────────────────── component pools ─────────────────────────────────

DOMAIN_NOUNS = [
    "bakery", "gym", "law firm", "dental clinic", "bookstore", "coffee shop",
    "car wash", "pet store", "yoga studio", "barbershop", "food truck",
    "auto repair shop", "flower shop", "nail salon", "tutoring center",
    "art gallery", "music school", "brewery", "winery", "farm stand",
    "daycare", "veterinary clinic", "physical therapy clinic", "spa",
    "tailor", "print shop", "bike shop", "hardware store", "grocery store",
    "pharmacy", "real estate agency", "travel agency", "insurance agency",
    "accounting firm", "marketing agency", "construction company",
    "landscaping company", "cleaning service", "moving company",
    "catering company", "event planner", "photography studio", "recording studio",
    "dance studio", "martial arts dojo", "climbing gym", "golf course",
    "ski resort", "campground", "marina", "food bank", "animal shelter",
    "community center", "co-working space", "laundromat", "comic shop",
    "game store", "thrift store", "butcher shop", "cheese shop", "tea house",
]

PLATFORM_SETS = [
    ["iOS", "Android"],
    ["Web"],
    ["iOS", "Android", "Web"],
    ["Web", "Cloud/Server"],
    ["Desktop"],
    ["Cloud/Server"],
    ["iOS", "Android", "Web", "Cloud/Server"],
]

# Stacks chosen to fit the surface set.
STACKS = [
    {"surfaces": {"iOS", "Android"}, "languages": ["Dart"],
     "frameworks": ["Flutter"], "databases": ["PostgreSQL", "SQLite"],
     "services": ["Firebase Auth", "Stripe"]},
    {"surfaces": {"Web"}, "languages": ["TypeScript"],
     "frameworks": ["React", "Node.js"], "databases": ["PostgreSQL"],
     "services": ["Auth0", "SendGrid"]},
    {"surfaces": {"Web"}, "languages": ["TypeScript"],
     "frameworks": ["Next.js", "NestJS"], "databases": ["PostgreSQL", "Redis"],
     "services": ["Clerk", "Stripe"]},
    {"surfaces": {"Cloud/Server"}, "languages": ["Python"],
     "frameworks": ["FastAPI"], "databases": ["PostgreSQL"],
     "services": ["Sentry", "S3"]},
    {"surfaces": {"Cloud/Server"}, "languages": ["TypeScript"],
     "frameworks": ["Node.js"], "databases": ["PostgreSQL", "Redis"],
     "services": ["voip.ms", "Twilio"]},
    {"surfaces": {"Desktop"}, "languages": ["Dart"],
     "frameworks": ["Flutter"], "databases": ["SQLite"],
     "services": ["Sentry"]},
    {"surfaces": {"iOS", "Android", "Web"}, "languages": ["Dart", "TypeScript"],
     "frameworks": ["Flutter", "Node.js"], "databases": ["PostgreSQL"],
     "services": ["Firebase Auth", "Stripe", "Sentry"]},
]

NAME_SUFFIXES = ["Hub", "Flow", "Go", "Now", "Pro", "Box", "Kit", "Desk",
                 "Mate", "Spot", "Link", "Wave", "Loop", "Pilot", "Nest",
                 "Stack", "Bridge", "Pulse", "Forge", "Scout"]

# App archetypes — each provides templated objectives/features/flow/edges/tasks.
# {d} = domain noun. Tasks use {lang}/{fw}/{db} for stack-specific instructions.
APP_TYPES = {
    "ordering": {
        "blurb": "lets {d} customers order ahead and track their order",
        "objectives": ["Online ordering", "Order tracking", "Menu management",
                       "Loyalty rewards", "Push notifications", "Scheduled pickup",
                       "Payments", "Reorder favorites"],
        "features": ["Menu browser", "Cart & checkout", "Order status",
                     "Rewards points", "Saved payment methods", "Pickup times"],
        "roles": ["customer", "staff member", "manager"],
        "flow": ["Open app", "Browse menu", "Add to cart", "Choose pickup time",
                 "Pay", "Track order", "Get ready notification"],
        "edges": [
            ("an item is sold out mid-order",
             "Tell the user and suggest a similar item without losing the cart.",
             "Handle sold-out item",
             "As a customer, I want a substitute suggested if an item sells out, "
             "so that I still complete my order.",
             "- Detect out-of-stock\n- Offer substitute\n- Preserve cart"),
            ("payment fails",
             "Show a clear error and let them retry or switch payment method.",
             "Recover from failed payment",
             "As a customer, I want to retry a failed payment, so that I don't "
             "lose my order.",
             "- Detect decline\n- Offer retry\n- Keep order draft"),
        ],
        "tasks": [
            ("Build menu browser screen",
             "Write a {fw} menu screen listing items from GET /menu. Objective: "
             "{lang} menu UI. Acceptance: items render, tap adds to cart. "
             "Verify: {fw} test passes.", "ui"),
            ("Create orders schema",
             "Objective: Create tables in {db}. Add menu_items, orders, "
             "order_items with FKs. Acceptance: migration applies. Verify: "
             "schema lists 3 tables.", "db"),
            ("Order status endpoint",
             "Objective: Write {lang} order-status API. GET /orders/:id/status "
             "returns live state. Acceptance: returns queued|making|ready. "
             "Verify: curl returns 200 with status.", "api"),
        ],
    },
    "booking": {
        "blurb": "lets people book appointments at a {d}",
        "objectives": ["Appointment booking", "Calendar sync", "Reminders",
                       "Staff scheduling", "Online payments", "Waitlist",
                       "Recurring bookings", "Cancellations"],
        "features": ["Availability calendar", "Booking form", "Reminder emails",
                     "Staff roster", "Deposit payments", "Reschedule flow"],
        "roles": ["client", "staff member", "owner"],
        "flow": ["Pick a service", "See availability", "Choose a slot",
                 "Enter details", "Pay deposit", "Get confirmation",
                 "Receive reminder"],
        "edges": [
            ("two clients book the same slot",
             "Lock the slot on selection and warn if it's already taken.",
             "Prevent double-booking",
             "As a client, I want to be told if my slot was just taken, so that I "
             "can pick another.",
             "- Lock slot on hold\n- Detect conflict\n- Offer next opening"),
            ("a client no-shows",
             "Flag no-shows and optionally charge the deposit.",
             "Handle no-shows",
             "As an owner, I want no-shows flagged, so that I can enforce my "
             "policy.",
             "- Mark no-show\n- Apply deposit rule\n- Notify client"),
        ],
        "tasks": [
            ("Build availability calendar",
             "Write a {fw} calendar showing open slots from GET /availability. "
             "Objective: {lang} calendar UI. Acceptance: open slots selectable. "
             "Verify: {fw} test passes.", "ui"),
            ("Create bookings schema",
             "Objective: Create tables in {db}. Add services, slots, bookings "
             "with a unique constraint on (slot_id). Acceptance: double-book "
             "fails. Verify: constraint rejects duplicate.", "db"),
            ("Booking confirmation service",
             "Objective: Write {lang} booking API. POST /bookings holds a slot "
             "and emails confirmation. Acceptance: returns 201, slot locked. "
             "Verify: test passes.", "api"),
        ],
    },
    "tracker": {
        "blurb": "helps {d} users log activity and track progress over time",
        "objectives": ["Activity logging", "Progress charts", "Goals",
                       "Reminders", "Streaks", "Export data", "Sharing",
                       "Trends"],
        "features": ["Quick log entry", "Progress graphs", "Goal setup",
                     "Reminder schedule", "Streak counter", "CSV export"],
        "roles": ["user", "coach", "admin"],
        "flow": ["Open dashboard", "Add a log entry", "See it on the chart",
                 "Check goal progress", "Get a reminder", "Review weekly trend"],
        "edges": [
            ("the user skips several days",
             "Keep streaks fair and nudge them back without shaming.",
             "Handle missed days",
             "As a user, I want a gentle nudge after missing days, so that I "
             "restart easily.",
             "- Detect gap\n- Preserve longest streak\n- Send nudge"),
            ("two devices log offline",
             "Merge offline entries without duplicates on sync.",
             "Merge offline logs",
             "As a user, I want offline entries merged cleanly, so that nothing "
             "is lost or doubled.",
             "- Queue offline\n- Dedup on sync\n- Resolve conflicts"),
        ],
        "tasks": [
            ("Build log entry screen",
             "Write a {fw} quick-log screen. Objective: {lang} entry UI. "
             "Acceptance: entry persists to API. Verify: {fw} test passes.", "ui"),
            ("Progress aggregation query",
             "Objective: Write {db} progress view aggregating per week. "
             "Acceptance: one row per week. Verify: query returns rows.", "db"),
            ("Streak service",
             "Objective: Write {lang} streak logic. Acceptance: gaps reset "
             "current, keep best. Verify: unit test passes.", "api"),
        ],
    },
    "marketplace": {
        "blurb": "is a marketplace connecting {d} buyers and sellers",
        "objectives": ["Listings", "Search & filters", "Messaging", "Checkout",
                       "Reviews", "Seller payouts", "Favorites", "Disputes"],
        "features": ["Listing browser", "Search filters", "Buyer-seller chat",
                     "Secure checkout", "Ratings & reviews", "Seller dashboard"],
        "roles": ["buyer", "seller", "admin"],
        "flow": ["Browse listings", "Filter results", "View a listing",
                 "Message the seller", "Checkout", "Leave a review"],
        "edges": [
            ("a listing is bought by two people at once",
             "Reserve on checkout start and release if abandoned.",
             "Prevent oversell",
             "As a buyer, I want exclusive checkout on a one-of item, so that I'm "
             "not charged for something already sold.",
             "- Reserve on checkout\n- Release on timeout\n- Block double-sale"),
            ("a buyer disputes a charge",
             "Open a dispute case and hold the seller payout.",
             "Handle disputes",
             "As a buyer, I want to open a dispute, so that I can recover a bad "
             "order.",
             "- Create case\n- Hold payout\n- Notify both parties"),
        ],
        "tasks": [
            ("Build listing browser",
             "Write a {fw} listing grid from GET /listings. Objective: {lang} "
             "grid UI. Acceptance: filters update results. Verify: {fw} test "
             "passes.", "ui"),
            ("Create listings + orders schema",
             "Objective: Create tables in {db} for listings, orders, reviews "
             "with FKs. Acceptance: migration applies. Verify: schema lists "
             "tables.", "db"),
            ("Checkout + reserve endpoint",
             "Objective: Write {lang} checkout API reserving an item. Acceptance: "
             "second checkout is blocked. Verify: test passes.", "api"),
        ],
    },
    "crm": {
        "blurb": "is a CRM for a {d} to manage contacts and deals",
        "objectives": ["Contact management", "Deal pipeline", "Reminders",
                       "Reporting", "Email integration", "Tasks", "Notes",
                       "Lead capture"],
        "features": ["Kanban pipeline", "Contact timeline", "Email sync",
                     "Dashboard reports", "Follow-up reminders", "Lead forms"],
        "roles": ["sales rep", "manager", "admin"],
        "flow": ["Open pipeline", "Drag a deal forward", "Open a contact",
                 "Log a call", "Set a follow-up", "Review the report"],
        "edges": [
            ("two reps edit the same deal",
             "Warn on version conflict instead of silently overwriting.",
             "Handle concurrent edits",
             "As a rep, I want a warning when a deal changed under me, so that I "
             "don't lose work.",
             "- Version column\n- Detect mismatch\n- Offer merge"),
            ("a lead form is spammed",
             "Add rate limiting and a captcha on the public form.",
             "Stop lead spam",
             "As an admin, I want spam blocked, so that the pipeline stays clean.",
             "- Rate limit\n- Captcha\n- Flag suspicious"),
        ],
        "tasks": [
            ("Build pipeline Kanban",
             "Write a {fw} Kanban moving deals across stages. Objective: {lang} "
             "board UI. Acceptance: drop updates stage. Verify: {fw} test "
             "passes.", "ui"),
            ("Create deals schema with versioning",
             "Objective: Create deals table in {db} with a version column. "
             "Acceptance: migration applies. Verify: column present.", "db"),
            ("Follow-up reminder job",
             "Objective: Write {lang} reminder cron emailing due follow-ups. "
             "Acceptance: fires at due time. Verify: test passes.", "api"),
        ],
    },
    "inventory": {
        "blurb": "helps a {d} scan stock and get low-stock alerts",
        "objectives": ["Barcode scanning", "Stock counts", "Low-stock alerts",
                       "Supplier orders", "Reports", "Multi-location",
                       "Audit log", "CSV export"],
        "features": ["Scanner", "Stock list", "Reorder alerts", "Supplier list",
                     "Count history", "CSV export"],
        "roles": ["staff member", "manager", "supplier"],
        "flow": ["Open scanner", "Scan a barcode", "Adjust the count",
                 "Save", "See the updated list", "Get a low-stock alert"],
        "edges": [
            ("an unknown barcode is scanned",
             "Prompt to add it as a new item with name and supplier.",
             "Add unknown item on scan",
             "As staff, I want to add an unknown item on scan, so that nothing is "
             "untracked.",
             "- Detect unknown\n- Show add form\n- Save with supplier"),
            ("the device is offline",
             "Queue changes locally and sync when back online.",
             "Work offline",
             "As staff, I want offline counts to sync later, so that I can work "
             "anywhere.",
             "- Queue offline\n- Sync on reconnect\n- Resolve conflicts"),
        ],
        "tasks": [
            ("Build scanner screen",
             "Write a {fw} barcode scanner resolving items. Objective: {lang} "
             "scanner UI. Acceptance: scan returns item or add-form. Verify: "
             "{fw} test passes.", "ui"),
            ("Create inventory schema",
             "Objective: Create tables in {db} for items, counts, suppliers. "
             "Acceptance: migration applies. Verify: schema lists tables.", "db"),
            ("Low-stock alert rule",
             "Objective: Write {lang} reorder alert under threshold. Acceptance: "
             "alert fires under min. Verify: unit test passes.", "api"),
        ],
    },
    "ivr": {
        "blurb": "is an AI phone system for a {d} that routes callers and can "
                 "call out",
        "objectives": ["Inbound routing", "Menu flows", "Voicemail",
                       "Outbound calls", "Call analytics", "Speech recognition",
                       "Call recording", "Transcripts"],
        "features": ["Visual flow builder", "Speech recognition", "Call recording",
                     "Analytics dashboard", "Voicemail inbox", "Outbound dialer"],
        "roles": ["caller", "agent", "admin"],
        "flow": ["Receive a call", "Play greeting", "Capture the choice",
                 "Match intent", "Route to a queue", "Log the outcome"],
        "edges": [
            ("the caller says something off-menu",
             "Fall back to an AI agent that understands free speech and routes by "
             "intent.",
             "Off-menu free-speech fallback",
             "As a caller, I want to just say what I need, so that I'm not stuck "
             "in a rigid menu.",
             "- Detect no match\n- Hand to AI agent\n- Route by intent"),
            ("all agents are busy",
             "Offer a callback or voicemail instead of holding forever.",
             "Offer callback when busy",
             "As a caller, I want a callback option, so that I don't wait on "
             "hold.",
             "- Detect full queue\n- Offer callback\n- Schedule it"),
        ],
        "tasks": [
            ("Inbound call webhook",
             "Write a {lang} webhook handling provider inbound events. "
             "Objective: {lang} call webhook. Acceptance: 200 + plays greeting. "
             "Verify: test passes.", "api"),
            ("Create calls + flows schema",
             "Objective: Create tables in {db} for calls and flows. Acceptance: "
             "migration applies. Verify: schema lists calls, flows.", "db"),
            ("Intent routing service",
             "Objective: Write {lang} intent router mapping speech to a queue. "
             "Acceptance: known phrases route correctly. Verify: test passes.",
             "api"),
        ],
    },
    "social": {
        "blurb": "is a community app for {d} enthusiasts to share and connect",
        "objectives": ["Posting", "Feed", "Following", "Likes & comments",
                       "Notifications", "Direct messages", "Groups", "Moderation"],
        "features": ["Post composer", "Home feed", "Follow system",
                     "Comments", "Notifications", "Direct messages"],
        "roles": ["member", "moderator", "admin"],
        "flow": ["Open feed", "Create a post", "See it in the feed",
                 "Get likes and comments", "Follow someone", "Get notified"],
        "edges": [
            ("a post is reported as abusive",
             "Queue it for moderation and hide it pending review.",
             "Moderate reported posts",
             "As a moderator, I want reported posts queued, so that I can keep "
             "the community safe.",
             "- Report action\n- Hide pending\n- Moderator queue"),
            ("the feed gets slow at scale",
             "Paginate and cache the feed.",
             "Scale the feed",
             "As a member, I want a fast feed, so that scrolling stays smooth.",
             "- Cursor pagination\n- Cache hot feed\n- Lazy media"),
        ],
        "tasks": [
            ("Build home feed",
             "Write a {fw} infinite feed from GET /feed. Objective: {lang} feed "
             "UI. Acceptance: scroll paginates. Verify: {fw} test passes.", "ui"),
            ("Create posts + follows schema",
             "Objective: Create tables in {db} for posts, follows, comments. "
             "Acceptance: migration applies. Verify: schema lists tables.", "db"),
            ("Notification fan-out",
             "Objective: Write {lang} notify service on new follower activity. "
             "Acceptance: followers receive events. Verify: test passes.", "api"),
        ],
    },
    "dashboard": {
        "blurb": "is an analytics dashboard for a {d}",
        "objectives": ["Reporting", "KPIs", "Charts", "Filters", "Exports",
                       "Alerts", "Scheduled reports", "Drilldown"],
        "features": ["KPI cards", "Trend charts", "Date filters", "CSV export",
                     "Threshold alerts", "Scheduled email reports"],
        "roles": ["analyst", "manager", "admin"],
        "flow": ["Open dashboard", "Pick a date range", "Read the KPIs",
                 "Drill into a chart", "Export the data", "Set an alert"],
        "edges": [
            ("a query is very slow",
             "Pre-aggregate into a materialized view and cache it.",
             "Speed up heavy reports",
             "As an analyst, I want fast reports, so that I'm not waiting on "
             "queries.",
             "- Materialized view\n- Refresh schedule\n- Cache results"),
            ("data is missing for a range",
             "Show a clear empty state instead of a broken chart.",
             "Handle missing data",
             "As an analyst, I want clear empty states, so that gaps aren't "
             "confusing.",
             "- Detect no rows\n- Show empty state\n- Suggest a range"),
        ],
        "tasks": [
            ("Build KPI dashboard",
             "Write a {fw} dashboard of KPI cards + charts. Objective: {lang} "
             "dashboard UI. Acceptance: cards reflect API data. Verify: {fw} "
             "test passes.", "ui"),
            ("Create metrics rollup",
             "Objective: Create a materialized view in {db} rolling up daily "
             "metrics. Acceptance: refresh populates it. Verify: view returns "
             "rows.", "db"),
            ("Threshold alert job",
             "Objective: Write {lang} alert job on KPI thresholds. Acceptance: "
             "alert fires past threshold. Verify: test passes.", "api"),
        ],
    },
    "lms": {
        "blurb": "is a learning platform for a {d} to deliver courses",
        "objectives": ["Courses", "Lessons", "Quizzes", "Progress tracking",
                       "Certificates", "Discussions", "Enrollment", "Grading"],
        "features": ["Course catalog", "Lesson player", "Quiz engine",
                     "Progress bar", "Certificates", "Discussion threads"],
        "roles": ["student", "instructor", "admin"],
        "flow": ["Browse courses", "Enroll", "Watch a lesson", "Take a quiz",
                 "See progress", "Earn a certificate"],
        "edges": [
            ("a student loses connection mid-quiz",
             "Auto-save answers and let them resume.",
             "Resume an interrupted quiz",
             "As a student, I want to resume a dropped quiz, so that I don't lose "
             "my answers.",
             "- Auto-save answers\n- Detect drop\n- Resume in place"),
            ("a quiz needs manual grading",
             "Route essay answers to the instructor's grading queue.",
             "Manual grading queue",
             "As an instructor, I want essays in a queue, so that I can grade "
             "them.",
             "- Detect manual items\n- Queue for grading\n- Post score"),
        ],
        "tasks": [
            ("Build lesson player",
             "Write a {fw} lesson player with progress. Objective: {lang} player "
             "UI. Acceptance: progress persists. Verify: {fw} test passes.",
             "ui"),
            ("Create courses schema",
             "Objective: Create tables in {db} for courses, lessons, enrollments, "
             "progress. Acceptance: migration applies. Verify: schema lists "
             "tables.", "db"),
            ("Quiz grading endpoint",
             "Objective: Write {lang} grading API. Acceptance: auto-grades MCQ, "
             "queues essays. Verify: test passes.", "api"),
        ],
    },
    "fintech": {
        "blurb": "helps {d} customers manage money and payments",
        "objectives": ["Accounts", "Transfers", "Budgeting", "Statements",
                       "Alerts", "Bill pay", "Categorization", "Goals"],
        "features": ["Account overview", "Transfer flow", "Budget tracker",
                     "Statement export", "Spending alerts", "Bill scheduler"],
        "roles": ["customer", "support agent", "admin"],
        "flow": ["See accounts", "Start a transfer", "Confirm with 2FA",
                 "See the new balance", "Categorize the spend", "Get an alert"],
        "edges": [
            ("a transfer is submitted twice",
             "Use an idempotency key so the same transfer never double-posts.",
             "Prevent duplicate transfers",
             "As a customer, I want a transfer to post once, so that I'm not "
             "charged twice.",
             "- Idempotency key\n- Detect retry\n- Return original result"),
            ("a login looks suspicious",
             "Step up to 2FA and alert the customer.",
             "Step-up auth on risk",
             "As a customer, I want extra checks on risky logins, so that my "
             "money is safe.",
             "- Risk score\n- Require 2FA\n- Notify customer"),
        ],
        "tasks": [
            ("Build accounts overview",
             "Write a {fw} accounts screen from GET /accounts. Objective: {lang} "
             "accounts UI. Acceptance: balances render. Verify: {fw} test "
             "passes.", "ui"),
            ("Create ledger schema",
             "Objective: Create double-entry ledger tables in {db}. Acceptance: "
             "debits equal credits constraint. Verify: imbalance rejected.", "db"),
            ("Idempotent transfer endpoint",
             "Objective: Write {lang} transfer API with idempotency keys. "
             "Acceptance: duplicate key returns the original. Verify: test "
             "passes.", "api"),
        ],
    },
    "helpdesk": {
        "blurb": "is a support helpdesk for a {d}",
        "objectives": ["Ticketing", "Queues", "SLAs", "Knowledge base",
                       "Macros", "Reporting", "Routing", "CSAT surveys"],
        "features": ["Ticket inbox", "Queue views", "SLA timers",
                     "Knowledge base", "Canned replies", "CSAT survey"],
        "roles": ["customer", "agent", "admin"],
        "flow": ["Submit a ticket", "Auto-route to a queue", "Agent replies",
                 "Track SLA", "Resolve", "Send CSAT survey"],
        "edges": [
            ("an SLA is about to breach",
             "Escalate and alert a lead before it breaches.",
             "Escalate before SLA breach",
             "As an admin, I want pre-breach escalation, so that SLAs are met.",
             "- Track SLA timer\n- Escalate near breach\n- Alert lead"),
            ("a ticket has no clear owner",
             "Round-robin assign within the right queue.",
             "Auto-assign tickets",
             "As an agent, I want tickets assigned fairly, so that none are "
             "dropped.",
             "- Detect unassigned\n- Round-robin\n- Notify assignee"),
        ],
        "tasks": [
            ("Build ticket inbox",
             "Write a {fw} ticket inbox from GET /tickets. Objective: {lang} "
             "inbox UI. Acceptance: queues filter tickets. Verify: {fw} test "
             "passes.", "ui"),
            ("Create tickets schema",
             "Objective: Create tables in {db} for tickets, queues, sla_events. "
             "Acceptance: migration applies. Verify: schema lists tables.", "db"),
            ("SLA escalation job",
             "Objective: Write {lang} SLA watcher escalating near breach. "
             "Acceptance: escalates before deadline. Verify: test passes.",
             "api"),
        ],
    },
    "logistics": {
        "blurb": "coordinates deliveries for a {d}",
        "objectives": ["Dispatch", "Route optimization", "Live tracking",
                       "Proof of delivery", "Driver app", "ETAs", "Zones",
                       "Notifications"],
        "features": ["Dispatch board", "Route map", "Live tracking",
                     "POD capture", "Driver app", "ETA notifications"],
        "roles": ["dispatcher", "driver", "customer"],
        "flow": ["Create a delivery", "Assign a driver", "Optimize the route",
                 "Track live", "Capture proof", "Notify the customer"],
        "edges": [
            ("a driver goes offline mid-route",
             "Reassign the remaining stops to the nearest driver.",
             "Reassign on driver dropout",
             "As a dispatcher, I want stops reassigned if a driver drops, so that "
             "deliveries still happen.",
             "- Detect offline\n- Find nearest driver\n- Reassign stops"),
            ("a customer isn't home",
             "Offer reschedule or safe-drop with a photo.",
             "Handle failed delivery",
             "As a customer, I want options if I miss a delivery, so that I still "
             "get my package.",
             "- Detect failed\n- Offer reschedule\n- Safe-drop photo"),
        ],
        "tasks": [
            ("Build dispatch board",
             "Write a {fw} dispatch board assigning deliveries. Objective: {lang} "
             "board UI. Acceptance: assign updates driver. Verify: {fw} test "
             "passes.", "ui"),
            ("Create deliveries schema",
             "Objective: Create tables in {db} for deliveries, stops, drivers. "
             "Acceptance: migration applies. Verify: schema lists tables.", "db"),
            ("Route optimization endpoint",
             "Objective: Write {lang} route API ordering stops. Acceptance: "
             "returns an ordered route. Verify: test passes.", "api"),
        ],
    },
    "cms": {
        "blurb": "is a content site for a {d} to publish and manage pages",
        "objectives": ["Content authoring", "Publishing", "Media library",
                       "SEO", "Scheduling", "Roles & permissions", "Versioning",
                       "Comments"],
        "features": ["Rich editor", "Publish workflow", "Media library",
                     "SEO settings", "Scheduled posts", "Role management"],
        "roles": ["author", "editor", "admin"],
        "flow": ["Create a draft", "Add media", "Set SEO", "Submit for review",
                 "Editor approves", "Publish"],
        "edges": [
            ("two authors edit one page",
             "Lock on edit and keep version history to recover from conflicts.",
             "Handle editing conflicts",
             "As an author, I want version history, so that conflicting edits "
             "don't lose work.",
             "- Edit lock\n- Version history\n- Restore a version"),
            ("a scheduled post fails to publish",
             "Retry and alert the author on repeated failure.",
             "Recover failed publish",
             "As an author, I want failed scheduled posts retried, so that "
             "content goes live.",
             "- Detect failure\n- Retry\n- Alert on give-up"),
        ],
        "tasks": [
            ("Build content editor",
             "Write a {fw} rich editor with draft save. Objective: {lang} editor "
             "UI. Acceptance: drafts persist. Verify: {fw} test passes.", "ui"),
            ("Create content schema with versions",
             "Objective: Create tables in {db} for pages, versions, media. "
             "Acceptance: migration applies. Verify: schema lists tables.", "db"),
            ("Publish workflow endpoint",
             "Objective: Write {lang} publish API with review states. Acceptance: "
             "draft→review→published transitions. Verify: test passes.", "api"),
        ],
    },
}

# First-message phrasings (request-phrasing diversity is a top quality lever).
FULL_TEMPLATES = [
    "I want to build {name}: an app that {blurb}. It runs on {plats}. It should "
    "{objs}. Key features: {feats}. Build it in {langs} with {fws}, {dbs} for "
    "storage, and {svcs}.",
    "Let's spec {name} — it {blurb}. Target platforms: {plats}. Goals: {objs}. "
    "Features I want: {feats}. Stack: {langs}/{fws}, {dbs}, plus {svcs}.",
    "{name} is an app that {blurb}, for {plats}. It needs to {objs} and include "
    "{feats}. Use {langs} with {fws} and {dbs}; integrate {svcs}.",
    "Building {name}: {blurb}. Runs on {plats}. Objectives: {objs}. Must-have "
    "features: {feats}. Tech: {langs}, {fws}, {dbs}, {svcs}.",
]
SPARSE_TEMPLATES = [
    "I want to build {name} — an app that {blurb}.",
    "Let's set up {name}. It {blurb}.",
    "Help me plan {name}: it {blurb}.",
    "{name} is a new app that {blurb}. Where do we start?",
]
OBJ_QUESTIONS = [
    "Objectives — anything else it should do?",
    "What are the main goals beyond what you mentioned?",
    "Which of these should it handle?",
    "What else is in scope for the first version?",
]
FEAT_QUESTIONS = [
    "Which features should it include?",
    "What features matter most for v1?",
    "Pick the features you want first.",
    "Which of these features are must-haves?",
]


# ───────────────────────── input evolution (Evol-Instruct-style) ───────────
# Raise USER-message entropy: inject varied real-world constraints and shift
# register/length. Combinatorial → far higher distinct-n + unique-message ratio,
# which is the lever against fast memorization / poor generalization.
CONSTRAINTS = [
    "for about {n} users", "on a tight budget", "we're a small team",
    "it needs to work offline", "launching next month", "for older users mostly",
    "it has to be fast", "with a clean simple design", "nothing fancy for now",
    "I need it ASAP", "this is a side project", "we'll scale to thousands later",
    "it must support Spanish too", "accessibility really matters",
    "mobile-first please", "we already have a logo", "keep it cheap to run",
    "must integrate with what we have", "privacy is a big deal here",
    "the team isn't technical", "we tried a no-code tool and outgrew it",
    "investors want a demo soon", "it should feel premium", "for a rural area",
    "we operate in 3 time zones", "mostly repeat customers", "high traffic on "
    "weekends", "I want analytics from day one", "offline-first is a must",
]
STYLE_PREFIX = ["", "", "", "hey, ", "ok so ", "quick one: ", "hi! ",
                "so basically ", "right, ", "thinking out loud — "]
STYLE_SUFFIX = ["", "", "", " thanks!", " any ideas?", " does that make sense?",
                " lmk", " 🙂", " — what do you think?", " thoughts?"]


def evolve_user(R, msg):
    """Apply a random constraint clause + light register/length shift."""
    out = msg
    if R.random() < 0.6:
        c = R.choice(CONSTRAINTS).replace("{n}", str(R.choice(
            [20, 50, 100, 200, 500, 1000])))
        joiner = R.choice([" — ", ", ", ". Also ", "; ", " and ", ". Oh and "])
        out = out.rstrip(".") + joiner + c + "."
    r = R.random()
    if r < 0.12:
        out = out.lower()                      # casual all-lowercase
    elif r < 0.18:
        out = out.replace("I want", "i wanna").replace("going to", "gonna")
    pre, suf = R.choice(STYLE_PREFIX), R.choice(STYLE_SUFFIX)
    if pre:
        out = pre + out[0].lower() + out[1:]
    return out + suf


def umsg(R, text):
    """A user turn with evolved phrasing (entropy)."""
    return user_msg(evolve_user(R, text))


# ───────────────────────── system prompts ──────────────────────────────────

def interview_system(p):
    topics = ("1. Industry — the domain [category: `industries`]\n"
              "2. Platforms — surfaces it runs on [category: `platforms`]\n"
              "3. Objectives — what it should do [category: `objectives`]\n"
              "4. Features — concrete features [category: `features`]\n"
              "5. Stack — languages/frameworks/databases/services "
              "[`languages`,`frameworks`,`databases`,`services`]")
    return f"""You are the Setup host for "{p['name']}" (Software Project). Your job is to build the project profile by TAGGING it. Keep every reply to 1-2 short sentences.

You have topics to fill (below). Tag what the user already told you FIRST, then ask about whatever is still open — one at a time, in flexible order.

{topics}
START FROM WHAT THEY SAID:
- Read the user's description and FIRST call `propose_tags` for everything it already implies.
- Then reflect back in one short sentence what you recorded.

HOW TO ASK (for the topics still open):
- Each remaining question goes through `ask_question` — it shows options as buttons the user taps.

RULES:
- Each tag VALUE is a SHORT label (≤5 words), one idea per tag; give several items as several tags.
- Once platforms are known, also propose at least one `languages` and one `frameworks` value yourself.
- When every required section has at least one tag, call `finalize_setup` (it refuses and lists what's missing if called too early)."""


def discovery_system(p):
    return f"""You are the project Coordinator running the post-setup DISCOVERY interview for "{p['name']}". Setup is done and NO tasks exist yet. Capture the FULL idea as a well-structured USER-STORY TREE before any work begins.

KEEP ASKING UNTIL COMPLETE
- Treat each answer as a starting point; keep interviewing until the whole flow is covered and the user says they're done.
- End every turn with exactly ONE focused question.

CAPTURE AS YOU GO
- Capture each distinct piece via `add_user_story` — a clear title and "As a <role>, I want <goal>, so that <benefit>", with acceptance_criteria when known.
- For a big chunk describing several things, call `draft_stories_from_text` with the raw words.

BUILD A REAL TREE
- One root epic; everything hangs under something meaningful.
- CHAIN flow steps: each step's `parent_story_id` is the step it follows from.
- `add_user_story` returns the new id — reuse it as the parent for its children."""


def pm_system(p):
    return f"""You are the Project Manager for "{p['name']}". You plan work and create/manage tasks for an autonomous software team. Decompose the plans and user stories into well-scoped tasks and assign each to the right agent.

Write each task as a concrete, stack-specific instruction using the chosen stack ({", ".join(p['languages'])} / {", ".join(p['frameworks'])}), with a clear objective, acceptance criteria, and a runnable verification command. Keep objectives as tiny imperative phrases (e.g. "Write {p['languages'][0]} login form", "Create users table in {p['databases'][0]}").

When the user asks to add work or break down plans — CALL THE TOOLS immediately (create_task, then assign_agent_to_task), then confirm what you changed in one short sentence."""


# ───────────────────────── scenario synthesis ──────────────────────────────

def make_scenario(R):
    # Start from a real industry + a NATURAL idea phrase (drives inference).
    industry = R.choice(list(INDUSTRIES.keys()))
    tax = INDUSTRIES[industry]
    idea = R.choice(tax["ideas"])
    # Pick an app archetype this industry implies (intersect with our templates).
    app_choices = [a for a in tax["apps"] if a in APP_TYPES] or list(APP_TYPES)
    app_key = R.choice(app_choices)
    app = APP_TYPES[app_key]
    domain = R.choice(DOMAIN_NOUNS)
    plats = R.choice(tax["platforms"])
    fits = [s for s in STACKS if s["surfaces"] & set(plats)] or STACKS
    stack = R.choice(fits)
    name = (idea.split()[-1] if idea.split()[-1].isalpha() else domain.split()[0])
    name = name.capitalize() + R.choice(NAME_SUFFIXES)

    # Objectives blend the industry's and the archetype's vocab (more coverage).
    obj_pool = list(dict.fromkeys(tax["objectives"] + app["objectives"]))
    objs = R.sample(obj_pool, R.randint(4, min(6, len(obj_pool))))
    feats = R.sample(app["features"], R.randint(3, min(5, len(app["features"]))))
    flow = app["flow"][:R.randint(4, min(6, len(app["flow"])))]
    edge = R.choice(app["edges"])
    role = R.choice(app["roles"])

    def fill(t):
        return (t.replace("{lang}", stack["languages"][0])
                 .replace("{fw}", stack["frameworks"][0])
                 .replace("{db}", stack["databases"][0])
                 .replace("{d}", domain))

    tasks = [{"title": fill(t), "description": fill(desc), "layer": layer}
             for (t, desc, layer) in app["tasks"]]

    return {
        "app": app_key, "name": name, "domain": domain, "idea": idea,
        "blurb": app["blurb"].format(d=domain),
        "industries": [industry],          # the GROUND TRUTH to infer from `idea`
        "platforms": plats,
        "objectives": objs, "features": feats,
        "languages": stack["languages"], "frameworks": stack["frameworks"],
        "databases": stack["databases"], "services": stack["services"],
        "role": role, "flow": flow, "edge": edge, "tasks": tasks,
        "epic_title": f"{app_key.capitalize()} for {domain}",
        "epic_narrative": f"As a {role}, I want to use {name}, so that the "
                          f"{domain} runs smoothly.",
    }


# ───────────────────────── tag helpers ─────────────────────────────────────

def _all_tags(p):
    return ([{"category": "industries", "value": v} for v in p["industries"]]
            + [{"category": "platforms", "value": v} for v in p["platforms"]]
            + [{"category": "objectives", "value": v} for v in p["objectives"]]
            + [{"category": "features", "value": v} for v in p["features"]]
            + [{"category": "languages", "value": v} for v in p["languages"]]
            + [{"category": "frameworks", "value": v} for v in p["frameworks"]]
            + [{"category": "databases", "value": v} for v in p["databases"]]
            + [{"category": "services", "value": v} for v in p["services"]])


def _stack_tags(p):
    return ([{"category": "languages", "value": v} for v in p["languages"]]
            + [{"category": "frameworks", "value": v} for v in p["frameworks"]]
            + [{"category": "databases", "value": v} for v in p["databases"]]
            + [{"category": "services", "value": v} for v in p["services"]])


def _plans_result():
    return {"ok": True, "plans": ["/PLANS/Overview.md", "/PLANS/Client.md",
                                  "/PLANS/Server.md", "/PLANS/Database.md"]}


def _fmt_list(xs):
    xs = [x.lower() for x in xs]
    return ", ".join(xs)


# ───────────────────────── conversation builders ───────────────────────────

def build_setup_full(p, R):
    ids = Ids()
    msg = R.choice(FULL_TEMPLATES).format(
        name=p["name"], blurb=p["blurb"], plats=", ".join(p["platforms"]),
        objs=_fmt_list(p["objectives"]), feats=_fmt_list(p["features"]),
        langs=", ".join(p["languages"]), fws=", ".join(p["frameworks"]),
        dbs=", ".join(p["databases"]), svcs=", ".join(p["services"]))
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, msg))]
    cid, call = tool_call(ids, "propose_tags", {"tags": _all_tags(p)})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(_all_tags(p))},
                            "propose_tags"))
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call], content=(
        f"You gave me everything — tagged the {p['industries'][0]} domain, "
        f"platforms, objectives, features, and a {p['languages'][0]}/"
        f"{p['frameworks'][0]} stack. Finalizing now.")))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — generated the Overview, Client, "
                          "Server, and Database plans. Ready for discovery."))
    return {"messages": msgs}


def build_setup_infer(p, R):
    """Pure INFERENCE: the user states a natural idea with NO industry/tags named
    (e.g. "I want to sell lemonade"); the host infers the industry + platforms +
    objectives and tags them, reflecting the inferred industry. This is what
    teaches "sell lemonade → Food & Beverage"."""
    ids = Ids()
    opener = R.choice(opener_for(p["idea"])).format(idea=p["idea"])
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, opener))]
    ind = p["industries"][0]
    inferred = ([{"category": "industries", "value": ind}]
                + [{"category": "platforms", "value": v} for v in p["platforms"]]
                + [{"category": "objectives", "value": v}
                   for v in p["objectives"][:R.randint(2, 3)]])
    cid, call = tool_call(ids, "propose_tags", {"tags": inferred})
    msgs.append(asst_calls([call],
                content=R.choice(INFER_REFLECTIONS).format(ind=ind)))
    msgs.append(tool_result(cid, {"ok": True, "added": len(inferred)},
                            "propose_tags"))
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(FEAT_QUESTIONS), "options": p["features"],
        "multi": True})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"answer": p["features"]}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "features", "value": v} for v in p["features"]]
                + _stack_tags(p)})
    msgs.append(asst_calls([call], content=(
        f"Adding those features and a {p['languages'][0]}/{p['frameworks'][0]} "
        f"stack.")))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated. Ready for "
                          "discovery."))
    return {"messages": msgs}


def build_setup_partial(p, R):
    ids = Ids()
    # Natural opener (no industry named) so this also teaches inference.
    msg = R.choice(opener_for(p["idea"])).format(idea=p["idea"])
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, msg))]
    seed = ([{"category": "industries", "value": v} for v in p["industries"]]
            + [{"category": "platforms", "value": v} for v in p["platforms"]]
            + [{"category": "objectives", "value": v}
               for v in p["objectives"][:2]])
    cid, call = tool_call(ids, "propose_tags", {"tags": seed})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(seed)}, "propose_tags"))
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(OBJ_QUESTIONS), "options": p["objectives"],
        "multi": True})
    msgs.append(asst_calls([call], content=(
        f"Recorded {p['industries'][0]} on {', '.join(p['platforms'])} — "
        f"now the rest.")))
    rest = p["objectives"][2:] or p["objectives"][:1]
    msgs.append(tool_result(cid, {"answer": rest}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "objectives", "value": v} for v in rest]})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(rest)}, "propose_tags"))
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(FEAT_QUESTIONS), "options": p["features"],
        "multi": True})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"answer": p["features"]}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "features", "value": v} for v in p["features"]]})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(p["features"])},
                            "propose_tags"))
    cid, call = tool_call(ids, "propose_tags", {"tags": _stack_tags(p)})
    msgs.append(asst_calls([call], content=(
        f"Proposing a {p['languages'][0]}/{p['frameworks'][0]} stack with "
        f"{p['databases'][0]}.")))
    msgs.append(tool_result(cid, {"ok": True, "added": len(_stack_tags(p))},
                            "propose_tags"))
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated. Ready for discovery."))
    return {"messages": msgs}


def build_setup_recovery(p, R):
    ids = Ids()
    msg = f"Let's set up {p['name']}. It's a {p['industries'][0]} app."
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, msg))]
    seed = ([{"category": "industries", "value": v} for v in p["industries"]]
            + [{"category": "platforms", "value": v} for v in p["platforms"]])
    cid, call = tool_call(ids, "propose_tags", {"tags": seed})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(seed)}, "propose_tags"))
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call]))
    missing = ["objectives", "features", "languages", "frameworks"]
    msgs.append(tool_result(cid, {"ok": False, "error": "not ready",
                "missing": missing,
                "message": "Add at least one tag to each: " + ", ".join(missing)},
                "finalize_setup"))
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(OBJ_QUESTIONS), "options": p["objectives"],
        "multi": True})
    msgs.append(asst_calls([call], content=(
        "Not ready yet — it still needs objectives and a stack. Let me gather "
        "those.")))
    msgs.append(tool_result(cid, {"answer": p["objectives"]}, "ask_question"))
    fill = ([{"category": "objectives", "value": v} for v in p["objectives"]]
            + [{"category": "features", "value": v} for v in p["features"]]
            + _stack_tags(p))
    cid, call = tool_call(ids, "propose_tags", {"tags": fill})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(fill)}, "propose_tags"))
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call], content="Everything's tagged now — finalizing."))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated. Ready for discovery."))
    return {"messages": msgs}


def build_setup_image(p, R):
    """User asks to see a mockup → generate_image, then continues setup."""
    ids = Ids()
    msg = (f"I'm building {p['name']} — {p['blurb']}. Can you show me a mockup "
           f"of the home screen first?")
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, msg))]
    cid, call = tool_call(ids, "generate_image", {
        "prompt": f"{p['app']} app home screen for a {p['domain']}, clean modern "
                  f"UI, {', '.join(p['platforms'])} style", "size": "1024x1792"})
    msgs.append(asst_calls([call], content="Here's a first look at the home "
                                           "screen."))
    msgs.append(tool_result(cid, {"ok": True,
                "image_url": "/workspace/img/mockup1.png"}, "generate_image"))
    seed = ([{"category": "industries", "value": v} for v in p["industries"]]
            + [{"category": "platforms", "value": v} for v in p["platforms"]]
            + [{"category": "objectives", "value": v} for v in p["objectives"][:3]])
    cid, call = tool_call(ids, "propose_tags", {"tags": seed})
    msgs.append(asst_calls([call], content="Tagging what you described while we "
                                           "look at it."))
    msgs.append(tool_result(cid, {"ok": True, "added": len(seed)}, "propose_tags"))
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(FEAT_QUESTIONS), "options": p["features"],
        "multi": True})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"answer": p["features"]}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "features", "value": v} for v in p["features"]]
                + _stack_tags(p)})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call], content="Looks good — finalizing the plan."))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated."))
    return {"messages": msgs}


def build_discovery(p, R):
    ids = Ids()
    opens = [
        f"I want a {p['role']} to {p['flow'][0].lower()} and go from there.",
        f"Let me walk you through how a {p['role']} uses it.",
        f"The core flow is what a {p['role']} does step by step.",
    ]
    msgs = [sys_msg(discovery_system(p)), user_msg(evolve_user(R, R.choice(opens)))]
    cid, call = tool_call(ids, "add_user_story", {
        "title": p["epic_title"], "narrative": p["epic_narrative"],
        "kind": "epic"})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"id": "S1"}, "add_user_story"))
    msgs.append(asst_calls([], content=None) if False else asst_text(
        f"Captured \"{p['epic_title']}\" as the root. Walk me through the first "
        f"thing a {p['role']} does."))
    chunk = (f"They {', '.join(s.lower() for s in p['flow'][:-1])}, and finally "
             f"{p['flow'][-1].lower()}.")
    msgs.append(user_msg(evolve_user(R, chunk)))
    use_draft = R.random() < 0.5
    if use_draft:
        cid, call = tool_call(ids, "draft_stories_from_text",
                              {"text": chunk, "parent_story_id": "S1"})
        msgs.append(asst_calls([call]))
        drafted = [{"id": f"S{i+2}", "title": s} for i, s in enumerate(p["flow"])]
        msgs.append(tool_result(cid, {"created": drafted},
                                "draft_stories_from_text"))
    else:
        # manual add, one story per step, batched
        calls = []
        for i, s in enumerate(p["flow"]):
            cid, call = tool_call(ids, "add_user_story", {
                "title": s,
                "narrative": f"As a {p['role']}, I want to {s.lower()}, so that "
                             f"the flow continues.",
                "parent_story_id": "S1", "kind": "story"})
            calls.append((cid, call, f"S{i+2}"))
        msgs.append(asst_calls([c for _, c, _ in calls]))
        for cid, _, sid in calls:
            msgs.append(tool_result(cid, {"id": sid}, "add_user_story"))
    # chain the steps
    chain = []
    prev = "S1"
    for i in range(len(p["flow"])):
        cid, call = tool_call(ids, "move_user_story", {
            "story_id": f"S{i+2}", "parent_story_id": prev, "order_index": 0})
        chain.append((cid, call, f"S{i+2}"))
        prev = f"S{i+2}"
    msgs.append(asst_calls([c for _, c, _ in chain]))
    for cid, _, sid in chain:
        msgs.append(tool_result(cid, {"ok": True, "story_id": sid},
                                "move_user_story"))
    case, answer, etitle, enarr, eaccept = p["edge"]
    msgs.append(asst_text(f"Chained the {len(p['flow'])} steps. What happens when "
                          f"{case}?"))
    msgs.append(user_msg(evolve_user(R, answer)))
    cid, call = tool_call(ids, "add_user_story", {
        "title": etitle, "narrative": enarr, "acceptance_criteria": eaccept,
        "parent_story_id": prev, "kind": "substory"})
    msgs.append(asst_calls([call]))
    edge_id = f"S{len(p['flow'])+2}"
    msgs.append(tool_result(cid, {"id": edge_id}, "add_user_story"))
    cid, call = tool_call(ids, "add_note", {
        "story_id": edge_id, "body": f"Edge case: {case}."})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True}, "add_note"))
    cid, call = tool_call(ids, "list_user_stories", {})
    msgs.append(asst_calls([call]))
    tree = [{"id": "S1", "title": p["epic_title"], "parent": None}] + [
        {"id": f"S{i+2}", "title": s} for i, s in enumerate(p["flow"])]
    msgs.append(tool_result(cid, {"stories": tree}, "list_user_stories"))
    msgs.append(asst_text("That captures the full flow with the edge case "
                          "handled. If nothing's missing, press \"Generate tasks "
                          "from stories\"."))
    return {"messages": msgs}


def build_tasks(p, R):
    ids = Ids()
    asks = [
        "Generate tasks from the setup and discovery stories.",
        "Break the plans down into tasks and assign them.",
        "Turn the user stories into concrete engineering tasks.",
        "Create the build tasks for v1 and assign each to an agent.",
    ]
    msgs = [sys_msg(pm_system(p)), user_msg(evolve_user(R, R.choice(asks)))]
    agent_for = {"ui": "sde-frontend", "db": "sde-database", "api": "sde-backend"}
    if "Flutter" in p["frameworks"]:
        agent_for["ui"] = "sde-flutter"
    calls, meta = [], []
    for t in p["tasks"]:
        cid, call = tool_call(ids, "create_task", {
            "title": t["title"], "description": t["description"],
            "priority": "HIGH" if t["layer"] in ("db", "ui") else "MED"})
        calls.append(call)
        meta.append((cid, t["layer"]))
    msgs.append(asst_calls(calls))
    tids = []
    for i, (cid, _) in enumerate(meta):
        msgs.append(tool_result(cid, {"id": f"T{i+1}", "status": "TODO"},
                                "create_task"))
        tids.append(f"T{i+1}")
    acalls = []
    for i, (_, layer) in enumerate(meta):
        cid, call = tool_call(ids, "assign_agent_to_task", {
            "task_id": tids[i], "agent_persona_id": agent_for.get(layer,
                                                                  "sde-backend")})
        acalls.append((cid, call))
    msgs.append(asst_calls([c for _, c in acalls]))
    for cid, _ in acalls:
        msgs.append(tool_result(cid, {"ok": True}, "assign_agent_to_task"))
    msgs.append(asst_text(
        f"Created {len(p['tasks'])} tasks from the stories and assigned each to "
        f"the right agent — every one has acceptance criteria and a verify "
        f"command."))
    return {"messages": msgs}


def build_tasks_breakdown(p, R):
    """Create a parent task, then sub-tasks under it (parent_task_id)."""
    ids = Ids()
    feature = R.choice(p["features"])
    msgs = [sys_msg(pm_system(p)),
            user_msg(evolve_user(R, f"Break down the \"{feature}\" feature into subtasks."))]
    cid, call = tool_call(ids, "create_task", {
        "title": f"Implement {feature}",
        "description": f"Objective: deliver {feature.lower()}. Parent task; "
                       f"split into UI, data, and API subtasks.",
        "priority": "HIGH"})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"id": "T1", "status": "TODO"}, "create_task"))
    subs, scalls = p["tasks"], []
    for t in subs:
        cid, call = tool_call(ids, "create_task", {
            "title": t["title"], "description": t["description"],
            "parent_task_id": "T1",
            "priority": "HIGH" if t["layer"] == "db" else "MED"})
        scalls.append((cid, call, t["layer"]))
    msgs.append(asst_calls([c for _, c, _ in scalls]))
    for i, (cid, _, _) in enumerate(scalls):
        msgs.append(tool_result(cid, {"id": f"T{i+2}", "status": "TODO"},
                                "create_task"))
    msgs.append(asst_text(f"Split \"{feature}\" into {len(subs)} subtasks under "
                          f"the parent. Ready to assign."))
    return {"messages": msgs}


def build_tasks_status(p, R):
    """List tasks then update a status — a maintenance interaction."""
    ids = Ids()
    msgs = [sys_msg(pm_system(p)),
            user_msg(evolve_user(R, "What is open, and mark the schema task in progress?"))]
    cid, call = tool_call(ids, "list_tasks", {})
    msgs.append(asst_calls([call]))
    open_tasks = [{"id": f"T{i+1}", "title": t["title"], "status": "TODO"}
                  for i, t in enumerate(p["tasks"])]
    msgs.append(tool_result(cid, {"tasks": open_tasks}, "list_tasks"))
    db_idx = next((i for i, t in enumerate(p["tasks"])
                   if t["layer"] == "db"), 0)
    cid, call = tool_call(ids, "update_task_status", {
        "task_id": f"T{db_idx+1}", "status": "IN_PROGRESS"})
    msgs.append(asst_calls([call], content=(
        f"{len(open_tasks)} tasks are open. Marking the schema task in "
        f"progress.")))
    msgs.append(tool_result(cid, {"ok": True, "status": "IN_PROGRESS"},
                            "update_task_status"))
    msgs.append(asst_text("Done — the schema task is now in progress."))
    return {"messages": msgs}


# ───────────────────────── orchestration ───────────────────────────────────

SETUP_VARIANTS = [build_setup_full, build_setup_partial, build_setup_recovery,
                  build_setup_image, build_setup_infer]
TASK_VARIANTS = [build_tasks, build_tasks_breakdown, build_tasks_status]


def generate(target, kinds, seed):
    R = random.Random(seed)
    convos = []
    # round-robin the kinds so the corpus stays balanced as it grows
    while len(convos) < target:
        p = make_scenario(R)
        if "setup" in kinds:
            for b in SETUP_VARIANTS:
                convos.append({**b(p, R), "tools": tools_for("setup")})
        if "discovery" in kinds:
            # two discovery variants (draft vs manual chosen inside)
            for _ in range(2):
                convos.append({**build_discovery(p, R),
                               "tools": tools_for("discovery")})
        if "tasks" in kinds:
            for b in TASK_VARIANTS:
                convos.append({**b(p, R), "tools": tools_for("tasks")})
    return convos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=10500,
                    help="how many conversations to synthesize (pre-dedupe)")
    ap.add_argument("--kinds", default="setup,discovery,tasks")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}

    convos = generate(args.target, kinds, args.seed)
    added, skipped = append_conversations(convos, source="generated")
    print(f"Synthesized {len(convos)} conversation(s) → added {added}, "
          f"skipped {skipped} dup. Kinds: {sorted(kinds)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
