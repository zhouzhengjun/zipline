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
from abc import (
    ABCMeta,
    abstractproperty,
    abstractmethod,
)

import pandas as pd
import numpy as np
from dateutil.relativedelta import MO, TH
from pandas import (
    DataFrame,
    date_range,
    DateOffset,
    DatetimeIndex,
)
from pandas.tseries.holiday import Holiday, nearest_workday, sunday_to_monday
from pandas.tseries.offsets import CustomBusinessDay, Day
from pandas.tseries.tools import normalize_date
from pandas.tslib import Timestamp
from six import with_metaclass

from zipline.errors import (
    InvalidCalendarName,
    CalendarNameCollision,
)
from zipline.utils.memoize import remember_last

from .calendar_helpers import (
    next_scheduled_day,
    previous_scheduled_day,
    next_open_and_close,
    previous_open_and_close,
    scheduled_day_distance,
    minutes_for_day,
    days_in_range,
    minutes_for_days_in_range,
    add_scheduled_days,
    next_scheduled_minute,
    previous_scheduled_minute,
)

start_default = pd.Timestamp('1990-01-01', tz='UTC')
end_base = pd.Timestamp('today', tz='UTC')
# Give an aggressive buffer for logic that needs to use the next trading
# day or minute.
end_default = end_base + pd.Timedelta(days=365)

NANOS_IN_MINUTE = 60000000000

MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = range(7)

USNewYearsDay = Holiday(
    'New Years Day',
    month=1,
    day=1,
    # When Jan 1 is a Sunday, US markets observe the subsequent Monday.
    # When Jan 1 is a Saturday (as in 2005 and 2011), no holiday is observed.
    observance=sunday_to_monday
)
USMartinLutherKingJrAfter1998 = Holiday(
    'Dr. Martin Luther King Jr. Day',
    month=1,
    day=1,
    # The US markets didn't observe MLK day as a holiday until 1998.
    start_date=Timestamp('1998-01-01'),
    offset=DateOffset(weekday=MO(3)),
)
USMemorialDay = Holiday(
    # NOTE: The definition for Memorial Day is incorrect as of pandas 0.16.0.
    # See https://github.com/pydata/pandas/issues/9760.
    'Memorial Day',
    month=5,
    day=25,
    offset=DateOffset(weekday=MO(1)),
)
USIndependenceDay = Holiday(
    'July 4th',
    month=7,
    day=4,
    observance=nearest_workday,
)
Christmas = Holiday(
    'Christmas',
    month=12,
    day=25,
    observance=nearest_workday,
)

MonTuesThursBeforeIndependenceDay = Holiday(
    # When July 4th is a Tuesday, Wednesday, or Friday, the previous day is a
    # half day.
    'Mondays, Tuesdays, and Thursdays Before Independence Day',
    month=7,
    day=3,
    days_of_week=(MONDAY, TUESDAY, THURSDAY),
    start_date=Timestamp("1995-01-01"),
)
FridayAfterIndependenceDayExcept2013 = Holiday(
    # When July 4th is a Thursday, the next day is a half day (except in 2013,
    # when, for no explicable reason, Wednesday was a half day instead).
    "Fridays after Independence Day that aren't in 2013",
    month=7,
    day=5,
    days_of_week=(FRIDAY,),
    observance=lambda dt: None if dt.year == 2013 else dt,
    start_date=Timestamp("1995-01-01"),
)
USBlackFridayBefore1993 = Holiday(
    'Black Friday',
    month=11,
    day=1,
    # Black Friday was not observed until 1992.
    start_date=Timestamp('1992-01-01'),
    end_date=Timestamp('1993-01-01'),
    offset=[DateOffset(weekday=TH(4)), Day(1)],
)
USBlackFridayInOrAfter1993 = Holiday(
    'Black Friday',
    month=11,
    day=1,
    start_date=Timestamp('1993-01-01'),
    offset=[DateOffset(weekday=TH(4)), Day(1)],
)


# http://en.wikipedia.org/wiki/Aftermath_of_the_September_11_attacks
September11Closings = date_range('2001-09-11', '2001-09-16', tz='UTC')

# http://en.wikipedia.org/wiki/Hurricane_sandy
HurricaneSandyClosings = date_range(
    '2012-10-29',
    '2012-10-30',
    tz='UTC'
)

# National Days of Mourning
# - President Richard Nixon - April 27, 1994
# - President Ronald W. Reagan - June 11, 2004
# - President Gerald R. Ford - Jan 2, 2007
USNationalDaysofMourning = [
    Timestamp('1994-04-27', tz='UTC'),
    Timestamp('2004-06-11', tz='UTC'),
    Timestamp('2007-01-02', tz='UTC'),
]


def days_at_time(days, t, tz, day_offset=0):
    """
    Shift an index of days to time t, interpreted in tz.

    Overwrites any existing tz info on the input.

    Parameters
    ----------
    days : DatetimeIndex
        The "base" time which we want to change.
    t : datetime.time
        The time we want to offset @days by
    tz : pytz.timezone
        The timezone which these times represent
    day_offset : int
        The number of days we want to offset @days by
    """
    days = DatetimeIndex(days).tz_localize(None).tz_localize(tz)
    days_offset = days + DateOffset(day_offset)
    return days_offset.shift(
        1, freq=DateOffset(hour=t.hour, minute=t.minute, second=t.second)
    ).tz_convert('UTC')


def holidays_at_time(calendar, start, end, time, tz):
    return days_at_time(
        calendar.holidays(
            # Workaround for https://github.com/pydata/pandas/issues/9825.
            start.tz_localize(None),
            end.tz_localize(None),
        ),
        time,
        tz=tz,
    )


def _overwrite_special_dates(midnight_utcs,
                             opens_or_closes,
                             special_opens_or_closes):
    """
    Overwrite dates in open_or_closes with corresponding dates in
    special_opens_or_closes, using midnight_utcs for alignment.
    """
    # Short circuit when nothing to apply.
    if not len(special_opens_or_closes):
        return

    len_m, len_oc = len(midnight_utcs), len(opens_or_closes)
    if len_m != len_oc:
        raise ValueError(
            "Found misaligned dates while building calendar.\n"
            "Expected midnight_utcs to be the same length as open_or_closes,\n"
            "but len(midnight_utcs)=%d, len(open_or_closes)=%d" % len_m, len_oc
        )

    # Find the array indices corresponding to each special date.
    indexer = midnight_utcs.get_indexer(special_opens_or_closes.normalize())

    # -1 indicates that no corresponding entry was found.  If any -1s are
    # present, then we have special dates that doesn't correspond to any
    # trading day.
    if -1 in indexer:
        bad_dates = list(special_opens_or_closes[indexer == -1])
        raise ValueError("Special dates %s are not trading days." % bad_dates)

    # NOTE: This is a slightly dirty hack.  We're in-place overwriting the
    # internal data of an Index, which is conceptually immutable.  Since we're
    # maintaining sorting, this should be ok, but this is a good place to
    # sanity check if things start going haywire with calendar computations.
    opens_or_closes.values[indexer] = special_opens_or_closes.values


class ExchangeCalendar(with_metaclass(ABCMeta)):
    """
    An ExchangeCalendar represents the timing information of a single market
    exchange.

    The timing information is made up of two parts: sessions, and opens/closes.

    A session represents a contiguous set of minutes and has a label, like
    "May 18". The label is purely a way to name a session and doesn't have any
    other meaning. We represent labels using UTC midnight timestamps, where
    only the year/month/day have any significance.

    Several methods in this class either take sessions or return sessions, and
    we attempt to be clear about this in their docstrings.  It's important to
    remember that these are sessions, and that the midnight UTC timestamp has
    no special significance as a moment in time.

    The opens and closes represent the UTC time of each session's open.
    """

    def __init__(self, start=start_default, end=end_default):
        tz = self.tz
        open_offset = self.open_offset
        close_offset = self.close_offset

        # Define those days on which the exchange is usually open.
        self.day = CustomBusinessDay(
            holidays=self.holidays_adhoc,
            calendar=self.holidays_calendar,
        )

        # Midnight in UTC for each trading day.
        _all_days = date_range(start, end, freq=self.day, tz='UTC')

        # `DatetimeIndex`s of standard opens/closes for each day.
        self._opens = days_at_time(_all_days, self.open_time, tz, open_offset)
        self._closes = days_at_time(
            _all_days, self.close_time, tz, close_offset
        )

        # `DatetimeIndex`s of nonstandard opens/closes
        _special_opens = self._special_opens(start, end)
        _special_closes = self._special_closes(start, end)

        # Overwrite the special opens and closes on top of the standard ones.
        _overwrite_special_dates(_all_days, self._opens, _special_opens)
        _overwrite_special_dates(_all_days, self._closes, _special_closes)

        # In pandas 0.16.1 _opens and _closes will lose their timezone
        # information. This looks like it has been resolved in 0.17.1.
        # http://pandas.pydata.org/pandas-docs/stable/whatsnew.html#datetime-with-tz  # noqa
        self.schedule = DataFrame(
            index=_all_days,
            columns=['market_open', 'market_close'],
            data={
                'market_open': self._opens,
                'market_close': self._closes,
            },
            dtype='datetime64[ns]',
        )

        self.first_trading_day = _all_days[0]
        self.last_trading_day = _all_days[-1]
        self.early_closes = DatetimeIndex(
            _special_closes.map(self.session_date)
        )

    def is_exchange_open(self, dt):
        """
        Given a dt, return whether this exchange is open at the given dt.

        Parameters
        ----------
        dt: pd.Timestamp
            The dt for which to check if this exchange is open.

        Returns
        -------
        bool
            Whether the exchange is open on this dt.
        """
        open_idx, close_idx = self._open_and_close_idx(dt)

        # FIXME probably need to do bounds checking on the first/last minute

        # if these indices are the same, it means the exchange is not open
        return open_idx != close_idx

    def next_open(self, dt):
        """
        Given a dt, returns the next open.

        Parameters
        ----------
        dt: pd.Timestamp
            The dt for which to get the next open.

        Returns
        -------
        pd.Timestamp
            The timestamp of the next open.
        """

        open_idx, _ = self._open_and_close_idx(dt)
        return self.schedule.market_open[open_idx]

    def next_close(self, dt):
        """
        Given a dt, returns the next close.

        Parameters
        ----------
        dt: pd.Timestamp
            The dt for which to get the next close.

        Returns
        -------
        pd.Timestamp
            The timestamp of the next close.
        """

        _, close_idx = self._open_and_close_idx(dt)
        return self.schedule.market_close[close_idx]

    def previous_open(self, dt):
        """
        Given a dt, returns the previous open.

        Parameters
        ----------
        dt: pd.Timestamp
            The dt for which to get the previous open.

        Returns
        -------
        pd.Timestamp
            The timestamp of the previous open.
        """
        open_idx, _ = self._open_and_close_idx(dt)
        return self.schedule.market_open[open_idx - 1]

    def previous_close(self, dt):
        """
        Given a dt, returns the previous close.

        Parameters
        ----------
        dt: pd.Timestamp
            The dt for which to get the previous close.

        Returns
        -------
        pd.Timestamp
            The timestamp of the previous close.
        """
        _, close_idx = self._open_and_close_idx(dt)
        return self.schedule.market_close[close_idx - 1]

    def next_exchange_minute(self, dt):
        """
        Given a dt, return the next exchange minute.

        Raises KeyError if the given timestamp is not an exchange minute.

        Parameters
        ----------
        dt: pd.Timestamp
            The dt for which to get the next exchange minute.

        Returns
        -------
        pd.Timestamp
            The next exchange minute.
        """

        # FIXME bounds check
        return self.all_trading_minutes[
            self.all_trading_minutes.get_loc(dt) + 1
        ]

    def previous_exchange_minute(self, dt):
        """
        Given a dt, return the previous exchange minute.

        Raises KeyError if the given timestamp is not an exchange minute.

        Parameters
        ----------
        dt: pd.Timestamp
            The dt for which to get the previous exchange minute.

        Returns
        -------
        pd.Timestamp
            The previous exchange minute.
        """

        # FIXME bounds check
        return self.all_trading_minutes[
            self.all_trading_minutes.get_loc(dt) - 1
        ]

    def session_date(self, dt):
        """
        Given an exchange minute, get the name of its containing session.

        Raises KeyError if the given minute is not an exchange minute.

        Parameters
        ----------
        dt : pd.Timestamp
            The dt for which to get the containing session.

        Returns
        -------
        pd.Timestamp
            The label of the containing session.  This is a UTC midnight
            timestamp.
        """
        open_idx, close_idx = self._open_and_close_idx(dt)

        if open_idx == close_idx:
            raise ValueError("Given dt {0} is not an exchange "
                             "minute!".format(dt))

        return self.schedule.index[close_idx]

    def next_session_date(self, session_date):
        """
        Given a session label, returns the label of the next session.

        Parameters
        ----------
        session_date: pd.Timestamp
            A session label (UTC midnight timestamp)

        Returns
        -------
        pd.Timestamp
            The label of the next session (UTC midnight timestamp)
        """
        session_date_idx = self.schedule.index.get_loc(session_date)
        return self.schedule.index[session_date_idx + 1]

    def previous_session_date(self, session_date):
        """
        Given a session label, returns the label of the previous session.

        Parameters
        ----------
        session_date: pd.Timestamp
            A session label (UTC midnight timestamp)

        Returns
        -------
        pd.Timestamp
            The label of the previous session (UTC midnight timestamp)
        """
        session_date_idx = self.schedule.index.get_loc(session_date)
        return self.schedule.index[session_date_idx - 1]

    def minutes_for_session(self, session_date):
        """
        Given a session label, return the minutes for that session.

        Parameters
        ----------
        session_date: pd.Timestamp
            A session label (UTC midnight timestamp)

        Returns
        -------
        pd.DateTimeIndex
            All the minutes for the given session.
        """
        session_data = self.schedule.loc[session_date]
        return self.all_trading_minutes[
            self.all_trading_minutes.slice_indexer(
                session_data.market_open,
                session_data.market_close
            )
        ]

    def exchange_sessions_in_range(self, start_session_date, end_session_date):
        """
        Given a start and end session date, return all the sessions in that
        range, inclusive.

        Parameters
        ----------
        start_session_date: The start of the desired range (UTC midnight
            timestamp)

        end_session_date: The end of the desired range (UTC midnight timestamp)

        Returns
        -------
        pd.DatetimeIndex
            The session dates in this range.
        """
        return self.all_trading_days[
            self.all_trading_days.slice_indexer(
                start_session_date,
                end_session_date
            )
        ]

    def _special_dates(self, calendars, ad_hoc_dates, start_date, end_date):
        """
        Union an iterable of pairs of the form

        (time, calendar)

        and an iterable of pairs of the form

        (time, [dates])

        (This is shared logic for computing special opens and special closes.)
        """
        _dates = DatetimeIndex([], tz='UTC').union_many(
            [
                holidays_at_time(calendar, start_date, end_date, time_,
                                 self.tz)
                for time_, calendar in calendars
            ] + [
                days_at_time(datetimes, time_, self.tz)
                for time_, datetimes in ad_hoc_dates
            ]
        )
        return _dates[(_dates >= start_date) & (_dates <= end_date)]

    def _special_opens(self, start, end):
        return self._special_dates(
            self.special_opens_calendars,
            self.special_opens_adhoc,
            start,
            end,
        )

    def _special_closes(self, start, end):
        return self._special_dates(
            self.special_closes_calendars,
            self.special_closes_adhoc,
            start,
            end,
        )

    @abstractproperty
    def name(self):
        """
        The name of this exchange calendar.
        E.g.: 'NYSE', 'LSE', 'CME Energy'
        """
        raise NotImplementedError()

    @abstractproperty
    def tz(self):
        """
        The native timezone of the exchange.

        SD: Not clear that this needs to be exposed.
        """
        raise NotImplementedError()

    def is_open_on_minute(self, dt):
        """
        Is the exchange open at minute @dt.

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

        # Retrieve the open and close for this exchange session
        _open, _close = self._get_open_and_close(session)

        # Is @dt within the trading hours for this exchange session
        return _open <= dt <= _close

    @abstractmethod
    def is_open_on_day(self, dt):
        """
        Is the exchange open anytime during @dt.

        SD: Need to decide whether this method answers the question:
        - Is exchange open at any time during the calendar day containing dt
        or
        - Is exchange open at any time during the trading session containg dt.
        Semantically it seems that the first makes more sense.

        Parameters
        ----------
        dt : Timestamp
            The UTC-canonicalized date.

        Returns
        -------
        bool
            True if exchange is open at any time during @dt.
        """
        raise NotImplementedError()

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
        start_session = self.session_date(start)
        end_session = self.session_date(end)
        # Increment end_session by one day, beucase .loc[s:e] return all values
        # in the DataFrame up to but not including `e`.
        # end_session += Timedelta(days=1)
        return self.schedule.loc[start_session:end_session]

    @property
    def all_trading_days(self):
        return self.schedule.index

    @property
    @remember_last
    def all_trading_minutes(self):
        opens_in_ns = \
            self._opens.values.astype('datetime64[ns]').astype(np.int64)

        closes_in_ns = \
            self._closes.values.astype('datetime64[ns]').astype(np.int64)

        deltas = closes_in_ns - opens_in_ns

        # + 1 because we want 390 days per standard day, not 389
        daily_sizes = (deltas / NANOS_IN_MINUTE) + 1
        num_minutes = np.sum(daily_sizes).astype(np.int64)

        # One allocation for the entire thing. This assumes that each day
        # represents a contiguous block of minutes, which might not always
        # be the case in the future.
        all_minutes = np.empty(num_minutes, dtype='datetime64[ns]')

        idx = 0
        for day_idx, size in enumerate(daily_sizes):
            # lots of small allocations, but it's fast enough for now.
            all_minutes[idx:(idx + size)] = \
                np.arange(
                    opens_in_ns[day_idx],
                    closes_in_ns[day_idx] + NANOS_IN_MINUTE,
                    NANOS_IN_MINUTE
                )

            idx += size

        return DatetimeIndex(all_minutes).tz_localize("UTC")

    def open_and_close(self, session_date):
        """
        Returns a tuple of timestamps of the open and close of the exchange
        session containing the given date. If the date is not in an exchange
        session, the next open and close are returned.

        Parameters
        ----------
        session_date: UTC Timestamp whose open and close are needed.

        Returns
        -------
        (Timestamp, Timestamp)
            The open and close for the given date.
        """
        o_and_c = self.schedule.loc[session_date]

        # `market_open` and `market_close` should be timezone aware, but pandas
        # 0.16.1 does not appear to support this:
        # http://pandas.pydata.org/pandas-docs/stable/whatsnew.html#datetime-with-tz  # noqa
        return (o_and_c['market_open'].tz_localize('UTC'),
                o_and_c['market_close'].tz_localize('UTC'))

    def session_date(self, dt):
        """
        Given a tz-aware time, returns the UTC-canonicalized date of the
        exchange session in which the time belongs. If the time is not in an
        exchange session (while the market is closed), returns the date of the
        next exchange session after the time.

        Parameters
        ----------
        dt : tz-aware Timestamp

        Returns
        -------
        Timestamp
            The date of the exchange session in which dt belongs.
        """
        open_idx, close_idx = self._open_and_close_idx(dt)

        return self.schedule.index[min(open_idx, close_idx)]

    def _open_and_close_idx(self, dt):
        open_idx = self.schedule.market_open.values.\
            astype('datetime64[ns]').searchsorted(np.datetime64(dt))
        close_idx = self.schedule.market_close.values.\
            astype('datetime64[ns]').searchsorted(np.datetime64(dt))

        return open_idx, close_idx


_static_calendars = {}


def get_calendar(name):
    """
    Retrieves an instance of an ExchangeCalendar whose name is given.

    Parameters
    ----------
    name : str
        The name of the ExchangeCalendar to be retrieved.
    """
    # First, check if the calendar is already registered
    if name not in _static_calendars:

        # Check if it is a lazy calendar. If so, build and register it.
        if name == 'NYSE':
            from zipline.utils.calendars.exchange_calendar_nyse \
                import NYSEExchangeCalendar
            nyse_cal = NYSEExchangeCalendar()
            register_calendar(nyse_cal)

        elif name == 'CME':
            from zipline.utils.calendars.exchange_calendar_cme \
                import CMEExchangeCalendar
            cme_cal = CMEExchangeCalendar()
            register_calendar(cme_cal)

        elif name == 'BMF':
            from zipline.utils.calendars.exchange_calendar_bmf \
                import BMFExchangeCalendar
            bmf_cal = BMFExchangeCalendar()
            register_calendar(bmf_cal)

        elif name == 'LSE':
            from zipline.utils.calendars.exchange_calendar_lse \
                import LSEExchangeCalendar
            lse_cal = LSEExchangeCalendar()
            register_calendar(lse_cal)

        elif name == 'TSX':
            from zipline.utils.calendars.exchange_calendar_tsx \
                import TSXExchangeCalendar
            tsx_cal = TSXExchangeCalendar()
            register_calendar(tsx_cal)

        else:
            # It's not a lazy calendar, so raise an exception
            raise InvalidCalendarName(calendar_name=name)

    return _static_calendars[name]


def deregister_calendar(cal_name):
    """
    If a calendar is registered with the given name, it is de-registered.

    Parameters
    ----------
    cal_name : str
        The name of the calendar to be deregistered.
    """
    try:
        _static_calendars.pop(cal_name)
    except KeyError:
        pass


def clear_calendars():
    """
    Deregisters all current registered calendars
    """
    _static_calendars.clear()


def register_calendar(calendar, force=False):
    """
    Registers a calendar for retrieval by the get_calendar method.

    Parameters
    ----------
    calendar : ExchangeCalendar
        The calendar to be registered for retrieval.
    force : bool, optional
        If True, old calendars will be overwritten on a name collision.
        If False, name collisions will raise an exception. Default: False.

    Raises
    ------
    CalendarNameCollision
        If a calendar is already registered with the given calendar's name.
    """
    # If we are forcing the registration, remove an existing calendar with the
    # same name.
    if force:
        deregister_calendar(calendar.name)

    # Check if we are already holding a calendar with the same name
    if calendar.name in _static_calendars:
        raise CalendarNameCollision(calendar_name=calendar.name)

    _static_calendars[calendar.name] = calendar
