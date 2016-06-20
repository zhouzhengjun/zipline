#
# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import time
from itertools import chain

from pandas import (
    Timedelta,
    Timestamp,
)
from pandas.tseries.holiday import(
    AbstractHolidayCalendar,
    Holiday,
)
from pytz import timezone

from .calendar_helpers import normalize_date

# Useful resources for making changes to this file:
# http://www.cmegroup.com/tools-information/holiday-calendar.html

from .exchange_calendar import (
    ExchangeCalendar,
    MONDAY,
    TUESDAY,
    WEDNESDAY,
    THURSDAY,
    USNewYearsDay,
    Christmas,
    FridayAfterIndependenceDayExcept2013,
    MonTuesThursBeforeIndependenceDay,
    USBlackFridayInOrAfter1993,
    September11Closings, HurricaneSandyClosings, USNationalDaysofMourning)

US_CENTRAL = timezone('America/Chicago')
CME_OPEN = time(17)
CME_CLOSE = time(16)

# The CME seems to have different holiday rules depending on the type
# of instrument.  For example, http://www.cmegroup.com/tools-information/holiday-calendar/files/2016-4th-of-july-holiday-schedule.pdf # noqa
# shows that Equity, Interest Rate, FX, Energy, Metals & DME Products close at
# 1200 CT on July 4, 2016, while Grain, Oilseed & MGEX Products and Livestock,
# Dairy & Lumber products are completely closed.

# For now, we will treat the CME as having a single calendar, and just go with
# the most conservative hours - and treat July 4 as an early close at noon.
CME_STANDARD_EARLY_CLOSE = time(12)

# Does the market open or close on a different calendar day, compared to the
# calendar day assigned by the exchange to this session?
CME_OPEN_OFFSET = -1
CME_CLOSE_OFFSET = -0

# These have the same definition, but are used in different places because the
# NYSE closed at 2:00 PM on Christmas Eve until 1993.
ChristmasEveBefore1993 = Holiday(
    'Christmas Eve',
    month=12,
    day=24,
    end_date=Timestamp('1993-01-01'),
    # When Christmas is a Saturday, the 24th is a full holiday.
    days_of_week=(MONDAY, TUESDAY, WEDNESDAY, THURSDAY),
)
ChristmasEveInOrAfter1993 = Holiday(
    'Christmas Eve',
    month=12,
    day=24,
    start_date=Timestamp('1993-01-01'),
    # When Christmas is a Saturday, the 24th is a full holiday.
    days_of_week=(MONDAY, TUESDAY, WEDNESDAY, THURSDAY),
)


class CMEHolidayCalendar(AbstractHolidayCalendar):
    """
    Non-trading days for the CME.

    See CMEExchangeCalendar for full description.
    """
    rules = [
        USNewYearsDay,
        Christmas,
    ]


class CMEEarlyCloseCalendar(AbstractHolidayCalendar):
    """
    Regular early close calendar for NYSE
    """
    # mlk day
    # presidents day
    # good friday
    # mem day
    # independence day
    # labor day
    # thanksgiving day

    rules = [
        MonTuesThursBeforeIndependenceDay,
        FridayAfterIndependenceDayExcept2013,
        USBlackFridayInOrAfter1993,
        ChristmasEveInOrAfter1993,
    ]


class CMEExchangeCalendar(ExchangeCalendar):
    """
    Exchange calendar for CME

    Open Time: 5:00 PM, America/Chicago
    Close Time: 5:00 PM, America/Chicago

    Regularly-Observed Holidays:
    - New Years Day (observed on monday when Jan 1 is a Sunday)
    - Martin Luther King Jr. Day (3rd Monday in January, only after 1998)
    - Washington's Birthday (aka President's Day, 3rd Monday in February)
    - Good Friday (two days before Easter Sunday)
    - Memorial Day (last Monday in May)
    - Independence Day (observed on the nearest weekday to July 4th)
    - Labor Day (first Monday in September)
    - Thanksgiving (fourth Thursday in November)
    - Christmas (observed on nearest weekday to December 25)

    NOTE: For the following US Federal Holidays, part of the CME is closed
    (Foreign Exchange, Interest Rates) but Commodities, GSCI, Weather & Real
    Estate is open.  Thus, we don't treat these as holidays.
    - Columbus Day
    - Veterans Day

    Regularly-Observed Early Closes:
    - Christmas Eve (except on Fridays, when the exchange is closed entirely)
    - Day After Thanksgiving (aka Black Friday, observed from 1992 onward)

    Additional Irregularities:
    - Closed from 9/11/2001 to 9/16/2001 due to terrorist attacks in NYC.
    - Closed on 10/29/2012 and 10/30/2012 due to Hurricane Sandy.
    - Closed on 4/27/1994 due to Richard Nixon's death.
    - Closed on 6/11/2004 due to Ronald Reagan's death.
    - Closed on 1/2/2007 due to Gerald Ford's death.
    - Closed at 1:00 PM on Wednesday, July 3rd, 2013
    - Closed at 1:00 PM on Friday, December 31, 1999
    - Closed at 1:00 PM on Friday, December 26, 1997
    - Closed at 1:00 PM on Friday, December 26, 2003

    NOTE: The exchange was **not** closed early on Friday December 26, 2008,
    nor was it closed on Friday December 26, 2014. The next Thursday Christmas
    will be in 2025.  If someone is still maintaining this code in 2025, then
    we've done alright...and we should check if it's a half day.
    """

    native_timezone = US_CENTRAL
    open_time = CME_OPEN
    close_time = CME_CLOSE
    open_offset = CME_OPEN_OFFSET
    close_offset = CME_CLOSE_OFFSET

    holidays_calendar = CMEHolidayCalendar()
    special_opens_calendars = ()
    special_closes_calendars = []

    holidays_adhoc = list(chain(
        September11Closings,
        HurricaneSandyClosings,
        USNationalDaysofMourning,
    ))

    special_opens_adhoc = ()
    special_closes_adhoc = []

    @property
    def name(self):
        return 'CME'

    @property
    def tz(self):
        return self.native_timezone

    def is_open_on_minute(self, dt):
        """
        Parameters
        ----------
        dt : Timestamp

        Returns
        -------
        bool
            True if  exchange is open at the given dt, otherwise False.
        """
        # Retrieve the exchange session relevant for this datetime
        session = self.session_date(dt)
        # Retrieve the opens and closes for this exchange session
        session_open, session_close = self.open_and_close(session)
        # Is @dt within the trading hours for this exchange session
        return (
            session_open and session_close and
            session_open <= dt <= session_close
        )

    def is_open_on_day(self, dt):
        """
        Is the exchange open (accepting orders) anytime during the calendar day
        containing @dt.

        Parameters
        ----------
        dt : Timestamp

        Returns
        -------
        bool
            True if  exchange is open at any time during the day containing @dt
        """
        dt_normalized = normalize_date(dt)
        return dt_normalized in self.schedule.index

    def trading_days(self, start, end):
        """
        Calculates all of the exchange sessions between the given
        start and end.

        SD: Presumably @start and @end are UTC-canonicalized, as our exchange
        sessions are. If not, then it's not clear how this method should behave
        if @start and @end are both in the middle of the day.

        Parameters
        ----------
        start : Timestamp
        end : Timestamp

        Returns
        -------
        DatetimeIndex
            A DatetimeIndex populated with all of the trading days between
            the given start and end.
        """
        return self.schedule.index[start:end]

    def open_and_close(self, dt):
        """
        Given a UTC-canonicalized date, returns a tuple of timestamps of the
        open and close of the exchange session on that date.

        SD: Can @date be an arbitrary datetime, or should we first map it to
        and exchange session using session_date. Need to check what the
        consumers expect. Here, I assume we need to map it to a session.

        Parameters
        ----------
        session : Timestamp
            The UTC-canonicalized session whose open and close are needed.

        Returns
        -------
        (Timestamp, Timestamp)
            The open and close for the given date.
        """
        session = self.session_date(dt)
        return self._get_open_and_close(session)

    def _get_open_and_close(self, session_date):
        """
        Retrieves the open and close for a given session.

        Parameters
        ----------
        session_date : Timestamp
            The canonicalized session_date whose open and close are needed.

        Returns
        -------
        (Timestamp, Timestamp) or (None, None)
            The open and close for the given dt, or Nones if the given date is
            not a session.
        """
        # Return a tuple of nones if the given date is not a session.
        if session_date not in self.schedule.index:
            return (None, None)

        o_and_c = self.schedule.loc[session_date]
        # `market_open` and `market_close` should be timezone aware, but pandas
        # 0.16.1 does not appear to support this:
        # http://pandas.pydata.org/pandas-docs/stable/whatsnew.html#datetime-with-tz  # noqa
        return (o_and_c['market_open'].tz_localize('UTC'),
                o_and_c['market_close'].tz_localize('UTC'))

    def session_date(self, dt):
        """
        Given a time, returns the UTC-canonicalized date of the exchange
        session in which the time belongs. If the time is not in an exchange
        session (while the market is closed), returns the date of the next
        exchange session after the time.

        Parameters
        ----------
        dt : Timestamp
            A timezone-aware Timestamp.

        Returns
        -------
        Timestamp
            The date of the exchange session in which dt belongs.
        """
        # Check if the dt is after the market close
        # If so, advance to the next day
        if self.is_open_on_day(dt):
            _, close = self._get_open_and_close(normalize_date(dt))
            if dt > close:
                dt += Timedelta(days=1)

        while not self.is_open_on_day(dt):
            dt += Timedelta(days=1)

        return normalize_date(dt)
