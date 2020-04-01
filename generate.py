#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
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
# SITE_URL = 'http://private/var/www/covid-estimator/html/'
CODE_BASE_PATH = os.path.dirname(os.path.realpath(__file__))
SITE_BASE_PATH =  os.path.join(CODE_BASE_PATH, 'html/')
JH_DATA_BASE_PATH = '/var/www/COVID-19/'
NYT_DATA_BASE_PATH = '/var/www/covid-19-data/'


def jh_date_to_dofy(date):
    """Convert a JH date string into a day of the year int."""
    return datetime.strptime(date, '%x').timetuple().tm_yday

def dofy_to_jh_date(dofy):
    """Convert day of the year into JH date."""
    return datetime.strptime('2020 {}'.format(dofy), '%Y %j').strftime('%x')

def nyt_date_to_dofy(date):
    """Convert a NYT date string into a day of the year int."""
    return datetime.strptime(date, '%Y-%m-%d').timetuple().tm_yday

def nyt_date_to_jh_date(date):
    """Convert a NYT date string into a JH date string."""
    return datetime.strptime(date, '%Y-%m-%d').strftime('%x')

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
    def estimate_order(self):
        """Returns quantifiable order of the estimation."""
        estimation_map = {
            'good': 0,
            'ok': 1,
            'bad': 2,
            'horrible': 3,
        }
        try:
            return estimation_map[self.estimate]
        except KeyError:
            return None

    @property
    def name(self):
        return '{}-{}'.format(self.country, self.province) if self.province else self.country

    @property
    def page_name(self):
        return '{}.html'.format(str(self.name).replace(' ', '-'))

    def get_url(self):
        return '{}{}'.format(SITE_URL, self.page_name)

    @property
    def current_cases_estimated(self):
        return int(self.cases_approximation.tail(1))

    @property
    def current_cases(self):
        return int(self.cases.tail(1))

    def to_chart_js(self):
        first_reported_case_index = self.cases.to_numpy().nonzero()[0][0]
        return {
            'id': "{}_data".format(self.id),
            'province': self.province,
            'country': self.country,
            'url': self.get_url(),
            'current_cases_estimated': self.current_cases_estimated,
            'current_cases': self.current_cases,
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
    states_population: Optional[pd.DataFrame] = None

    def __init__(self, jh_base_path='./', nyt_base_path='./', mortality_rate=3.4, avg_days_symptoms_to_death=14, avg_days_to_symptoms=5):
        """
        :param jh_base_path: Path to the Johns Hopkins' data repo https://github.com/CSSEGISandData/COVID-19
        :param mortality_rate: Median mortality rate in %. https://www.who.int/dg/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---3-march-2020
        :param avg_days_symptoms_to_death: Median days from symptoms to death. https://www.worldometers.info/coronavirus/coronavirus-death-rate/#days
        :param avg_days_to_symptoms: Median days from infected to symptoms. https://www.ncbi.nlm.nih.gov/pubmed/32150748
        """
        self.jh_base_path = jh_base_path
        self.nyt_base_path = nyt_base_path
        self.mortality_rate = mortality_rate
        self.avg_days_symptoms_to_death = avg_days_symptoms_to_death
        self.avg_days_to_death = avg_days_to_symptoms + avg_days_symptoms_to_death

        # Load states population data.
        self.states_population = pd.read_csv(
            os.path.join(CODE_BASE_PATH, 'statepop.tsv'),
            delimiter='\t',
        )


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

    def load_data_jh(self, deaths_file_path, cases_file_path):
        """
        Load data from Johns Hopkins' data repo into the memory.
        :return: self
        """

        # Load Confirmed Cases
        self.df_cases = pd.read_csv(
            os.path.join(
                self.jh_base_path,
                cases_file_path
            )
        )

        # Load deaths
        df = pd.read_csv(
            os.path.join(
                self.jh_base_path,
                deaths_file_path
            )
        )
        self.df_deaths = df.fillna(0)  # Fill nan with 0

        return self

    def load_data_nyt(self, states_file_path):
        """
        Load data from Johns Hopkins' data repo into the memory.
        :return: self
        """

        # Load raw states data.
        nyt_states_data = pd.read_csv(
            os.path.join(
                self.nyt_base_path,
                states_file_path
            )
        )

        last_data_day = nyt_date_to_dofy(nyt_states_data.tail(1)['date'].iloc[0])

        df_cases = pd.DataFrame(
            columns=['Province/State', 'Country/Region', 'Lat', 'Long']+[dofy_to_jh_date(x) for x in range(1, last_data_day)]
        )

        df_deaths = df_cases.copy()

        for i, row in nyt_states_data.iterrows():
            date_jh_format = nyt_date_to_jh_date(row['date'])
            state = str(row['state'])
            cases = int(row['cases'])
            deaths = int(row['deaths'])

            # just to make sure that we have the column with this date.
            # In case the initial data is not properly ordered.
            if date_jh_format not in df_cases.columns:
                df_cases[date_jh_format] = 0
                df_deaths[date_jh_format] = 0

            # Alter the existing row or make a new entry for this state.
            if len(df_cases.loc[(df_cases['Province/State'] == state) & (df_cases['Country/Region'] == 'US'), date_jh_format]):
                df_cases.loc[(df_cases['Province/State'] == state) & (df_cases['Country/Region'] == 'US'), date_jh_format] = cases
            else:
                df_cases = df_cases.append(
                    pd.Series([state, 'US', cases], index=['Province/State', 'Country/Region', date_jh_format]),
                    ignore_index=True,
                )

            # Alter the existing row or make a new entry for this state.
            if len(df_deaths.loc[(df_deaths['Province/State'] == state) & (df_deaths['Country/Region'] == 'US'), date_jh_format]):
                df_deaths.loc[(df_deaths['Province/State'] == state) & (df_deaths['Country/Region'] == 'US'), date_jh_format] = deaths
            else:
                df_deaths = df_deaths.append(
                    pd.Series([state, 'US', deaths], index=['Province/State', 'Country/Region', date_jh_format]),
                    ignore_index=True,
                )

        # Fill nan with 0
        self.df_cases = df_cases.fillna(0)
        self.df_deaths = df_deaths.fillna(0)

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
            raise ValueError('Data has not been loaded properly. Call load_data_jh() method first.')

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
                row_days = row_days.apply(jh_date_to_dofy)     # Convert dates into days of the year.

                # Get real cases in the same format.
                if row_province:
                    cases = self.df_cases.loc[
                            (self.df_cases['Province/State'] == row_province) & (self.df_cases['Country/Region'] == row_country)    # Sort by the same country and province.
                        ].iloc[0].iloc[4:]   # [1] gets the Series out of next() results, iloc[4:] cuts the unused meta data from the beginning.
                else:
                    cases = self.df_cases.loc[
                            self.df_cases['Country/Region'] == row_country    # Sort by the same country and province.
                        ].iloc[0].iloc[4:]   # [1] gets the Series out of next() results, iloc[4:] cuts the unused meta data from the beginning.

                # If we have less than 20 cases or 7 days of data or if the reported death rate is above 10%,
                # we cannot really estimate anything.
                if (
                        deaths_row.tail(1)[0] < 20
                        or deaths_row.tail(7)[4] < 1
                        or (deaths_row.sum() / cases.sum() * 100) > 10
                ):
                    continue

                population = 500000
                if row_country == 'US' and row_province in self.states_population['state'].tolist():
                    population = int(self.states_population.loc[self.states_population['state'] == row_province]['population'])

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
                    maxfev=population,
                )
                fit_a = fit[0][0]
                error_a = np.sqrt(fit[1][0][0])
                fit_peak_days = fit[0][1]
                error_peak_days = np.sqrt(fit[1][1][1])
                fit_c = fit[0][2]
                error_cases = np.sqrt(fit[1][2][2])

                # Make an estimated row with day # as values and dates as indexes.
                cases_approximation = pd.Series(row_days, index=deaths_row.index)\
                    .apply(
                        lambda x: int(self.logistic_model((x+self.avg_days_to_death), fit_a, fit_peak_days, fit_c)/self.mortality_rate*100)
                    )

                # Just a quick estimate.
                tail_cases_doubling_2d = float(deaths_row.tail(1))/self.mortality_rate*100*2**(self.avg_days_to_death/2)
                tail_cases_doubling_4d = float(deaths_row.tail(1))/self.mortality_rate*100*2**(self.avg_days_to_death/4)
                tail_cases_doubling_6d = float(deaths_row.tail(1))/self.mortality_rate*100*2**(self.avg_days_to_death/6)

                # Estimated cases as of today.
                tail_cases_approximation = int(cases_approximation.tail(1))

                # If the estimation shows that there are more cases than an estimation with 2-day doubling rate, it's likely bad.
                if error_a > 3 or error_peak_days > 150 or error_cases > population * 0.01 or tail_cases_approximation > tail_cases_doubling_2d:
                    estimate = 'bad'
                    # Skipp cases with not specified quality.
                    if 'bad' not in quality:
                        continue
                elif error_a > 1 or error_peak_days > 50 or error_cases > population * 0.001:
                    # Skipp cases with not specified quality.
                    if 'ok' not in quality:
                        continue
                    estimate = 'ok'
                else:
                    # Skipp cases with not specified quality.
                    if 'good' not in quality:
                        continue
                    estimate = 'good'

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
        data = sorted(data, key=lambda x: (x.estimate_order, x.error_cases))

        return data


def save_graph(data):
    """
    Saves an png graph of provided data into a specified file.
    :param data: DataEstimate object
    :return:
    """
    # Cut blank data.
    cases_approximation = data.cases_approximation.tail(len(data.cases_approximation) - data.cases_approximation.to_numpy().nonzero()[0][0])

    cases_approximation = cases_approximation.copy()

    filename = '{}.png'.format(data.name)
    plot = cases_approximation.plot(
        title='{} estimated vs. {} reported cases in {}{}'.format(
            '{:,}'.format(data.current_cases_estimated),
            '{:,}'.format(data.current_cases),
            data.country,
            ' - {}'.format(data.province) if data.province else '',
        ),
        kind='bar',
        color='g',
        legend=None,
        use_index=False,
        figsize=(19, 10),
    )
    figure = plot.get_figure()
    figure.savefig(
        os.path.join(
            SITE_BASE_PATH,
            filename,
        ),
        format='png',
        bbox_inches='tight',
        optimize=True
    )
    plt.close(figure)
    return filename


def render_page(data, filename, social_image_filename='social.jpg'):
    """
    Render a single HTML page with given data.
    :param data: List of DataEstimate objects.
    :param filename: Filename to be generated, with extension and no path. e.g. index.html
    :return:
    """
    file_loader = FileSystemLoader(os.path.join(CODE_BASE_PATH, 'templates'))
    env = Environment(loader=file_loader)
    template = env.get_template('index.html')

    output = template.render(
        data=[x.to_chart_js() for x in data],
        site_url=SITE_URL,
        date_updated=datetime.now().strftime('%B %d'),
        debug=DEBUG,
        filename=filename,
        social_image_filename=social_image_filename,
    )
    with open(os.path.join(SITE_BASE_PATH, filename), 'w') as file:
        file.write(output)


def render_pages():
    """Render world pages from JH data."""
    covid_analyzer = CovidDataAnalyzer(jh_base_path=JH_DATA_BASE_PATH, nyt_base_path=NYT_DATA_BASE_PATH)
    data = covid_analyzer.load_data_jh(
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

        # NOTICE: Not being used at the moment as JH has not been updating the US data.
        # Save US related data.
        if data_row.country == 'US':
            data_us.append(data_row)

        # Save world related data.
        if data_row.country != 'US':
            # Render specific graphs and pages.
            social_image_filename = save_graph(data_row)
            render_page([data_row], '{}'.format(data_row.page_name), social_image_filename=social_image_filename)

            # Save world data for world.html
            data_world.append(data_row)

    # Render the world page.
    render_page(data_world, 'world.html')

    """Render US pages from NYT data."""
    covid_analyzer = CovidDataAnalyzer(jh_base_path=JH_DATA_BASE_PATH, nyt_base_path=NYT_DATA_BASE_PATH)
    data = covid_analyzer.load_data_nyt(states_file_path='us-states.csv').approximate_data(
        include_countries=['All'],
        exclude_countries=[],
        quality=['good', 'ok', ]
    )

    data_us = []
    # Iterate over each region.
    for data_row in data:

        # Save US related data.
        if data_row.country == 'US':
            # Render specific graphs and pages.
            social_image_filename = save_graph(data_row)
            render_page([data_row], '{}'.format(data_row.page_name), social_image_filename=social_image_filename)

            data_us.append(data_row)

    # Render home page. (US)
    render_page(data_us, 'index.html')


# Run page generator.
if __name__ == "__main__":
    render_pages()
