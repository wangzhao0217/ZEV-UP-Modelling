"""Purpose vocabulary extracted without loading the legacy plotting stack."""
from enum import Enum

class TripPurpose(Enum):
    """Enumeration of the 14 trip purposes with standardized names"""
    COMMUTING = "Commuting"
    SHOPPING = "Shopping"
    VISITING_FRIENDS_RELATIVES = "Visiting friends or relatives"
    EDUCATION = "Education"
    SPORT_ENTERTAINMENT = "Sport/entertainment"
    OTHER_PERSONAL_BUSINESS = "Other personal business"
    EATING_DRINKING = "Eating/drinking"
    OTHER_JOURNEY = "Other journey"
    BUSINESS = "Business"
    HEALTH_VISITS = "Health visits"
    ESCORT = "Escort"
    HOLIDAY_DAYTRIP = "Holiday/daytrip"
    RECREATION = "Recreation"
    SOCIAL_VISITS = "Social visits"
