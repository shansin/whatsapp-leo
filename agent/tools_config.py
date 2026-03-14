"""Tool selection configuration for each MCP server.

TOOL_CONFIG maps server name → allowlist (or None for all tools).

  None          expose every tool the server provides
  [list]        expose only the named tools

Edit the lists below to control what the LLM can call.
Run `scripts/update_tools_config.py` after adding a new MCP server to
auto-populate its entry and keep existing lists in sync.
"""

TOOL_CONFIG: dict[str, list[str] | None] = {
    "brave": None,

    "playwright": [
        "browser_navigate",                     # Navigate to a URL
        "browser_navigate_back",                # Go back to the previous page in the history
        #"browser_click",                        # Perform click on a web page
        #"browser_type",                         # Type text into editable element
        #"browser_fill_form",                    # Fill multiple form fields
        #"browser_press_key",                    # Press a key on the keyboard
        #"browser_take_screenshot",              # Take a screenshot of the current page. You can't perform actions based on the screenshot, use browser_snapshot for actions.
        "browser_snapshot",                     # Capture accessibility snapshot of the current page, this is better than screenshot
        #"browser_select_option",                # Select an option in a dropdown
        #"browser_tabs",                         # List, create, close, or select a browser tab.
        #"browser_wait_for",                     # Wait for text to appear or disappear or a specified time to pass
        #"browser_evaluate",                     # Evaluate JavaScript expression on page or element
        #"browser_close",                        # Close the page
        #"browser_console_messages",             # Returns all console messages
        #"browser_drag",                         # Perform drag and drop between two elements
        #"browser_file_upload",                  # Upload one or multiple files
        #"browser_handle_dialog",                # Handle a dialog
        #"browser_hover",                        # Hover over element on page
        #"browser_install",                      # Install the browser specified in the config. Call this if you get an error about the browser not being installed.
        #"browser_network_requests",             # Returns all network requests since loading the page
        #"browser_resize",                       # Resize the browser window
        #"browser_run_code",                     # Run Playwright code snippet
    ],

    "workspace": [
        #"docs.find",                            # Finds Google Docs by searching for a query in their title. Supports pagination.
        #"docs.getText",                         # Retrieves the text content of a Google Doc.
        #"docs.create",                          # Creates a new Google Doc. Can be blank or with Markdown content.
        #"docs.appendText",                      # Appends text to the end of a Google Doc.
        #"docs.replaceText",                     # Replaces all occurrences of a given text with new text in a Google Doc.
        #"docs.insertText",                      # Inserts text at the beginning of a Google Doc.
        #"docs.move",                            # Moves a document to a specified folder.
        #"drive.search",                         # Searches for files and folders in Google Drive. The query can be a simple search term, a Google Drive URL, or a full query string. For more information on query strings see: https://developers.google.com/drive/api/guides/search-files
        #"drive.findFolder",                     # Finds a folder by name in Google Drive.
        #"drive.createFolder",                   # Creates a new folder in Google Drive.
        #"drive.downloadFile",                   # Downloads the content of a file from Google Drive to a local path. Note: Google Docs, Sheets, and Slides require specialized handling.
        "calendar.list",                        # Lists all of the user's calendars.
        "calendar.listEvents",                  # Lists events from a calendar. Defaults to upcoming events.
        "calendar.getEvent",                    # Gets the details of a specific calendar event.
        "calendar.createEvent",                 # Creates a new event in a calendar.
        "calendar.updateEvent",                 # Updates an existing event in a calendar.
        "calendar.deleteEvent",                 # Deletes an event from a calendar.
        "calendar.findFreeTime",                # Finds a free time slot for multiple people to meet.
        "calendar.respondToEvent",              # Responds to a meeting invitation (accept, decline, or tentative).
        #"sheets.getText",                       # Retrieves the content of a Google Sheets spreadsheet.
        #"sheets.getRange",                      # Gets values from a specific range in a Google Sheets spreadsheet.
        #"sheets.find",                          # Finds Google Sheets spreadsheets by searching for a query. Supports pagination.
        "gmail.search",                         # Search for emails in Gmail using query parameters.
        "gmail.get",                            # Get the full content of a specific email message.
        #"gmail.send",                           # Send an email message.
        "gmail.createDraft",                    # Create a draft email message.
        #"gmail.sendDraft",                      # Send a previously created draft email.
        #"gmail.modify",                         # Modify a Gmail message. Supported modifications include:
        #"people.getMe",                         # Gets the profile information of the authenticated user.
        #"time.getCurrentDate",                  # Gets the current date. Returns both UTC (for calendar/API use) and local time (for display to the user), along with the timezone.
        #"time.getCurrentTime",                  # Gets the current time. Returns both UTC (for calendar/API use) and local time (for display to the user), along with the timezone.
        "auth.clear",                           # Clears the authentication credentials, forcing a re-login on the next request.
        "auth.refreshToken",                    # Manually triggers the token refresh process.
        #"chat.findDmByEmail",                   # Finds a Google Chat DM space by a user's email address.
        #"chat.findSpaceByName",                 # Finds a Google Chat space by its display name.
        #"chat.getMessages",                     # Gets messages from a Google Chat space.
        #"chat.listSpaces",                      # Lists the spaces the user is a member of.
        #"chat.listThreads",                     # Lists threads from a Google Chat space in reverse chronological order.
        #"chat.sendDm",                          # Sends a direct message to a user.
        #"chat.sendMessage",                     # Sends a message to a Google Chat space.
        #"chat.setUpSpace",                      # Sets up a new Google Chat space with a display name and a list of members.
        #"docs.extractIdFromUrl",                # Extracts the document ID from a Google Workspace URL.
        #"gmail.createLabel",                    # Create a new Gmail label. Labels help organize emails into categories.
        "gmail.downloadAttachment",             # Downloads an attachment from a Gmail message to a local file.
        #"gmail.listLabels",                     # List all Gmail labels in the user's mailbox.
        #"people.getUserProfile",                # Gets a user's profile information.
        #"people.getUserRelations",              # Gets a user's relations (e.g., manager, spouse, assistant, etc.). Common relation types include: manager, assistant, spouse, partner, relative, mother, father, parent, sibling, child, friend, domesticPartner, referredBy. Defaults to the authenticated user if no userId is provided.
        #"sheets.getMetadata",                   # Gets metadata about a Google Sheets spreadsheet.
        #"slides.find",                          # Finds Google Slides presentations by searching for a query. Supports pagination.
        #"slides.getImages",                     # Downloads all images embedded in a Google Slides presentation to a local directory.
        #"slides.getMetadata",                   # Gets metadata about a Google Slides presentation.
        #"slides.getSlideThumbnail",             # Downloads a thumbnail image for a specific slide in a Google Slides presentation to a local path.
        #"slides.getText",                       # Retrieves the text content of a Google Slides presentation.
        #"time.getTimeZone",                     # Gets the local timezone. Note: timezone is also included in getCurrentDate and getCurrentTime responses.
    ],

    "garmin": [
        "get_stats",                            # Get daily activity stats with curated essential metrics
        "get_user_summary",                     # Get user summary data (compatible with garminconnect-ha)
        "get_stats_and_body",                   # Get stats and body composition data
        "get_sleep_data",                       # Get full sleep data with all details
        "get_sleep_summary",                    # Get sleep summary with only essential metrics (lightweight version)
        "get_heart_rates",                      # Get full heart rate time-series data
        "get_heart_rates_summary",              # Get heart rate summary with essential metrics (lightweight version)
        "get_hrv_data",                         # Get Heart Rate Variability (HRV) data
        "get_rhr_day",                          # Get resting heart rate data
        "get_stress_data",                      # Get full stress time-series data
        "get_stress_summary",                   # Get stress summary with essential metrics (lightweight version)
        "get_body_battery",                     # Get body battery data with events
        "get_training_readiness",               # Get training readiness data with curated metrics
        "get_training_status",                  # Get training status with curated metrics
        "get_activities",                       # Get activities with pagination support
        "get_activity",                         # Get basic activity information
        "get_activities_by_date",               # Get activities data between specified dates, optionally filtered by activity type
        "count_activities",                     # Get total count of activities in the user's Garmin account
        "get_steps_data",                       # Get detailed steps data with 15-minute intervals
        "get_daily_steps",                      # Get steps data for a date range
        "get_weekly_steps",                     # Get weekly step data aggregates
        "get_floors",                           # Get floors climbed data
        "get_body_composition",                 # Get body composition data for a single date or date range
        "get_weigh_ins",                        # Get weight measurements between specified dates
        #"add_weigh_in",                         # Add a new weight measurement
        "get_training_effect",                  # Get training effect data for a specific activity
        "get_endurance_score",                  # Get endurance score data between dates
        "get_hill_score",                       # Get hill score data between dates
        "get_lactate_threshold",                # Get lactate threshold data
        "get_personal_record",                  # Get personal records for user
        #"get_nutrition_daily_food_log",         # Get daily food consumption records for a date
        #"log_food",                             # Log a food item to a specific meal on a date
        #"add_body_composition",                 # Add body composition data
        #"add_gear_to_activity",                 # Associate gear with an activity
        #"add_hydration_data",                   # Add hydration data
        #"add_weigh_in_with_timestamps",         # Add a new weight measurement with specific timestamps
        #"create_custom_food",                   # Create a custom food in the user's Garmin nutrition library
        #"delete_weigh_ins",                     # Delete weight measurements for a specific date
        #"delete_workout",                       # Delete a workout from Garmin Connect
        #"download_workout",                     # Download a workout as a FIT file
        #"get_activities_fordate",               # Get activities for a specific date
        #"get_activity_exercise_sets",           # Get exercise sets for strength training activities
        #"get_activity_gear",                    # Get gear data used for an activity
        "get_activity_hr_in_timezones",         # Get heart rate data in different time zones for an activity
        "get_activity_split_summaries",         # Get split summaries for an activity
        "get_activity_splits",                  # Get splits for an activity
        "get_activity_typed_splits",            # Get typed splits for an activity
        "get_activity_types",                   # Get all available activity types
        "get_activity_weather",                 # Get weather data for an activity
        #"get_adhoc_challenges",                 # Get user-created social/group challenges (e.g., step competitions with friends)
        "get_all_day_events",                   # Get daily wellness events data
        "get_all_day_stress",                   # Get all-day stress data
        #"get_available_badge_challenges",       # Get official Garmin badge challenges available to join
        #"get_badge_challenges",                 # Get all badge challenges the user has joined (completed and in-progress)
        #"get_blood_pressure",                   # Get blood pressure data
        "get_body_battery_events",              # Get body battery events data
        #"get_custom_food_serving_units",        # Get available serving units for custom foods
        #"get_custom_foods",                     # Search or list user's custom foods
        #"get_daily_weigh_ins",                  # Get weight measurements for a specific date
        #"get_device_alarms",                    # Get alarms from all Garmin devices
        #"get_device_last_used",                 # Get information about the last used Garmin device
        #"get_device_settings",                  # Get settings for a specific Garmin device
        #"get_device_solar_data",                # Get solar data for a specific device
        #"get_devices",                          # Get all Garmin devices associated with the user account
        #"get_earned_badges",                    # Get earned badges for user
        "get_fitnessage_data",                  # Get fitness age data
        #"get_full_name",                        # Get user's full name from profile
        #"get_gear",                             # Get all gear registered with the user account
        #"get_goals",                            # Get Garmin Connect goals (active, future, or past)
        #"get_hydration_data",                   # Get hydration data
        #"get_inprogress_virtual_challenges",    # Get in-progress virtual challenges/expeditions
        #"get_lifestyle_logging_data",           # Get lifestyle logging data for a specific date
        #"get_menstrual_calendar_data",          # Get menstrual calendar data between specified dates
        #"get_menstrual_data_for_date",          # Get menstrual data for a specific date
        #"get_morning_training_readiness",       # Get morning training readiness score
        #"get_non_completed_badge_challenges",   # Get badge challenges currently in progress (not yet completed)
        #"get_nutrition_daily_meals",            # Get daily meal summaries for a date
        #"get_nutrition_daily_settings",         # Get nutrition plan/settings for a date
        #"get_pregnancy_summary",                # Get pregnancy summary data
        #"get_primary_training_device",          # Get information about the primary training device
        "get_progress_summary_between_dates",   # Get progress summary for a metric between dates
        "get_race_predictions",                 # Get predicted race times based on current fitness level
        "get_respiration_data",                 # Get full respiration time-series data
        "get_respiration_summary",              # Get respiration summary with essential metrics (lightweight version)
        "get_scheduled_workouts",               # Get scheduled workouts between two dates with curated summary list
        #"get_spo2_data",                        # Get SpO2 (blood oxygen) data
        "get_training_plan_workouts",           # Get training plan workouts for the week containing the given date
        #"get_unit_system",                      # Get user's preferred unit system from profile
        "get_user_profile",                     # Get user profile information
        #"get_userprofile_settings",             # Get user profile settings
        "get_weekly_intensity_minutes",         # Get weekly intensity minutes data aggregates
        "get_weekly_stress",                    # Get weekly stress data aggregates
        "get_workout_by_id",                    # Get detailed information for a specific workout
        "get_workouts",                         # Get all workouts with curated summary list
        #"remove_gear_from_activity",            # Remove gear association from an activity
        #"request_reload",                       # Request reload of epoch data
        #"schedule_workout",                     # Schedule a workout to a specific calendar date
        #"set_blood_pressure",                   # Set blood pressure values
        #"update_custom_food",                   # Update an existing custom food in the user's Garmin nutrition library
        #"upload_workout",                       # Upload a workout from JSON data
    ],
}


def make_tool_filter(server_name: str) -> dict | None:
    """Return a static tool_filter dict for MCPServerStdio, or None for all tools."""
    allowed = TOOL_CONFIG.get(server_name)
    if allowed is None:
        return None
    return {"allowed_tool_names": allowed}
