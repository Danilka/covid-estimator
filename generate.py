#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from typing import Optional

from jinja2 import FileSystemLoader, Environment
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

__author__ = 'Danil Kozyatnikov'
__copyright__ = None
__license__ = ''
__version__ = '0.1'
__email__ = 'covidestimator@danilink.com'

DEBUG = False
SITE_URL = 'https://coronaviruspredictor.com/'
SITE_BASE_PATH = '/var/www/covid-estimator/html/'
DATA_BASE_PATH = '/var/www/COVID-19/'


def date_to_dofy(date):
    """Convert a date string into a day of the year int."""
    return datetime.strptime(date, '%x').timetuple().tm_yday


def date_to_readable(date):
    """Convert a date string into a readable format."""
    return datetime.strptime(date, '%x').date().strftime('%b %d')


class DataEstimate:

    id = None   # type: Optional[int]
    province = None     # type: Optional[str]
    country = None  # type: Optional[str]
    cases_approximation = None  # type: Optional[pd.Series]
    cases = None    # type: Optional[pd.Series]
    estimate = None     # type: Optional[str]
    error_cases = None  # type: Optional[float]
    error_peak = None   # type: Optional[float]
    peak_day = None    # type: Optional[int]

    def __init__(
        self,
        id: Optional[int] = None,
        province: Optional[str] = None,
        country: Optional[str] = None,
        cases_approximation: Optional[pd.Series] = None,
        cases: Optional[pd.Series] = None,
        estimate: Optional[str] = None,
        error_cases: Optional[float] = None,
        error_peak: Optional[float] = None,
        peak_day: Optional[int] = None,
    ):
        self.id = id
        self.province = province
        self.country = country
        self.cases_approximation = cases_approximation
        self.cases = cases
        self.estimate = estimate
        self.error_cases = error_cases
        self.error_peak = error_peak
        self.peak_day = peak_day

    @property
    def name(self):
        return '{}-{}'.format(self.country, self.province) if self.province else self.country

    @property
    def page_name(self):
        return '{}.html'.format(str(self.name).replace(' ', '-'))

    def get_url(self):
        return '{}{}'.format(SITE_URL, self.page_name)

    def to_chart_js(self):
        first_reported_case_index = self.cases.to_numpy().nonzero()[0][0]
        return {
            'id': "{}_data".format(self.id),
            'province': self.province,
            'country': self.country,
            'url': self.get_url(),
            'current_cases_estimated': int(self.cases_approximation.tail(1)),
            'current_cases': int(self.cases.tail(1)),
            'last_day_reported': date_to_readable(self.cases.tail(1).index[0]),
            'cases_a': str(list(self.cases_approximation.iloc[first_reported_case_index:])),
            'cases': str(list(self.cases.iloc[first_reported_case_index:])),
            'estimate': self.estimate,
            'error_cases': int(self.error_cases) if self.error_cases > 1 else 1,
            'error_peak': int(self.error_peak) if self.error_peak > 1 else False,
            'peak_day': datetime.strptime('{} {}'.format(2020+int(int(self.peak_day/365)), int(self.peak_day%365)), '%Y %j').date().strftime('%B %d'),
            'labels': str(
                list(
                    [
                        date_to_readable(x) for x in list(
                            self.cases_approximation.iloc[first_reported_case_index:].index
                        )
                    ]
                )
            ),
        }


class CovidDataAnalyzer:

    df_cases: Optional[pd.DataFrame] = None
    df_deaths: Optional[pd.DataFrame] = None

    def __init__(self, base_path='./', mortality_rate=3.4, avg_days_symptoms_to_death=14, avg_days_to_symptoms=5):
        """
        :param base_path: Path to the Johns Hopkins' data repo https://github.com/CSSEGISandData/COVID-19
        :param mortality_rate: Median mortality rate in %. https://www.who.int/dg/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---3-march-2020
        :param avg_days_symptoms_to_death: Median days from symptoms to death. https://www.worldometers.info/coronavirus/coronavirus-death-rate/#days
        :param avg_days_to_symptoms: Median days from infected to symptoms. https://www.ncbi.nlm.nih.gov/pubmed/32150748
        """
        self.base_path = base_path
        self.mortality_rate = mortality_rate
        self.avg_days_symptoms_to_death = avg_days_symptoms_to_death
        self.avg_days_to_death = avg_days_to_symptoms + avg_days_symptoms_to_death

    @staticmethod
    def logistic_model(day, infection_speed, maximum, total):
        """
        Logistic model function.
        :param day: # of the day tha is being calculated.
        :param infection_speed: Speed of infection.
        :param maximum: The day when maximum infections occurs.
        :param total: Total number of infected by the end.
        :return:
        """
        return total/(1+np.exp(-(day-maximum)/infection_speed))

    def load_data(self, deaths_file_path, cases_file_path):
        """
        Load data from Johns Hopkins' data repo into the memory.
        :return: self
        """

        # Load Confirmed Cases
        self.df_cases = pd.read_csv(
            os.path.join(
                self.base_path,
                cases_file_path
            )
        )

        # Load deaths
        df = pd.read_csv(
            os.path.join(
                self.base_path,
                deaths_file_path
            )
        )
        self.df_deaths = df.fillna(0)  # Fill nan with 0

        return self

    def approximate_data(
            self,
            include_countries=('All', ),
            exclude_countries=('All', ),
            include_province=('All',),
            exclude_province=('All',),
            quality=('good', 'ok'),
    ):
        if not len(self.df_cases) or not len(self.df_deaths):
            raise ValueError('Data has not been loaded properly. Call load_data() method first.')

        data = []   # type: [DataEstimate]

        for i, deaths_row in self.df_deaths.iterrows():
            try:
                # row = df.iloc[0]    # Get one row of data.
                row_province = deaths_row[0]
                row_country = deaths_row[1]

                # Filter by country.
                if 'All' in include_countries:
                    if row_country in exclude_countries:
                        continue
                elif row_country not in include_countries:
                    continue

                # Filter by region.
                if 'All' in include_province:
                    if row_province in exclude_province:
                        continue
                elif row_province not in include_province:
                    continue

                deaths_row = deaths_row.iloc[4:]  # Cut meta data at the beginning.

                row_days = deaths_row.index.to_series()    # Split dates info a separate series.
                row_days = row_days.apply(date_to_dofy)     # Convert dates into days of the year.

                # Get real cases in the same format.
                if row_province:
                    cases = self.df_cases.loc[
                            (self.df_cases['Province/State'] == row_province) & (self.df_cases['Country/Region'] == row_country)    # Sort by the same country and province.
                        ].iloc[0].iloc[4:]   # [1] gets the Series out of next() results, iloc[4:] cuts the unused meta data from the beginning.
                else:
                    cases = self.df_cases.loc[
                            self.df_cases['Country/Region'] == row_country    # Sort by the same country and province.
                        ].iloc[0].iloc[4:]   # [1] gets the Series out of next() results, iloc[4:] cuts the unused meta data from the beginning.

                # If we have less than 20 cases or 5 days of data or if the reported death rate is above 10%,
                # we cannot really estimate anything.
                if (
                        deaths_row.tail(1)[0] < 10
                        or deaths_row.tail(5)[4] < 1
                        or (deaths_row.sum() / cases.sum() * 100) > 10
                ):
                    continue

                # Find the model fit.
                fit = curve_fit(
                    self.logistic_model,
                    list(row_days.iloc[deaths_row.to_numpy().nonzero()[0][0]:]),
                    list(deaths_row.iloc[deaths_row.to_numpy().nonzero()[0][0]:]),
                    # p0=[2, 10, 200000],
                    bounds=(
                        (1, 1, 1e4),
                        (200, 1000, 3e7),
                    ),
                    check_finite=True,
                    maxfev=100000
                )
                fit_a = fit[0][0]
                error_a = np.sqrt(fit[1][0][0])
                fit_peak_days = fit[0][1]
                error_peak_days = np.sqrt(fit[1][1][1])
                fit_c = fit[0][2]
                error_cases = np.sqrt(fit[1][2][2])

                if error_a > 3 or error_peak_days > 150 or error_cases > 1e4:
                    estimate = 'bad'
                    # Skipp cases with not specified quality.
                    if 'bad' not in quality:
                        continue
                elif error_a > 1 or error_peak_days > 50 or error_cases > 1e3:
                    # Skipp cases with not specified quality.
                    if 'ok' not in quality:
                        continue
                    estimate = 'ok'
                else:
                    # Skipp cases with not specified quality.
                    if 'good' not in quality:
                        continue
                    estimate = 'good'

                # Make an estimated row with day # as values and dates as indexes.
                cases_approximation = pd.Series(row_days, index=deaths_row.index)\
                    .apply(
                        lambda x: int(self.logistic_model((x+self.avg_days_to_death), fit_a, fit_peak_days, fit_c)/self.mortality_rate*100)
                    )

                data.append(
                    DataEstimate(
                        id=i,
                        province=row_province,
                        country=row_country,
                        cases_approximation=cases_approximation,
                        cases=cases,
                        estimate=estimate,
                        error_cases=error_cases,
                        error_peak=error_peak_days,
                        peak_day=fit_peak_days,
                    )
                )

            except IndexError:
                # Some states don't have any data yet, so we just skipp them.
                continue

        # Sort data by fitment.
        data = sorted(data, key=lambda x: x.error_cases)

        return data


def render_page(data, filename):
    """
    Render a single HTML page with given data.
    :param data: List of DataEstimate objects in
    :param filename: Filename to be generated, with extension and no path. e.g. index.html
    :return:
    """
    file_loader = FileSystemLoader('templates')
    env = Environment(loader=file_loader)
    template = env.get_template('index.html')

    output = template.render(
        data=[x.to_chart_js() for x in data],
        site_url=SITE_URL,
        date_updated=datetime.now().strftime('%B %d'),
        debug=DEBUG,
    )
    with open(os.path.join(SITE_BASE_PATH, filename), 'w') as file:
        file.write(output)


def render_pages():
    """Render all site pages."""

    covid_analyzer = CovidDataAnalyzer(base_path=DATA_BASE_PATH)
    data = covid_analyzer.load_data(
        cases_file_path='csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
        deaths_file_path='csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
    ).approximate_data(
        include_countries=['All'],
        exclude_countries=[],
        quality=['good', 'ok', ]
    )

    data_world = []
    data_us = []
    # Iterate over each region.
    for data_row in data:

        # Render specific pages.
        render_page([data_row], '{}'.format(data_row.page_name))

        # Save US related data.
        if data_row.country == 'US':
            data_us.append(data_row)

        # Save world related data.
        if data_row.country != 'US':
            data_world.append(data_row)

    # Render home page. (US)
    render_page(data_us, 'index.html')

    # Render the world page.
    render_page(data_world, 'world.html')


# Run page generator.
render_pages()
