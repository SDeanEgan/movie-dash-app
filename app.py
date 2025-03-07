from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import urllib.request
import json
from bs4 import BeautifulSoup
import numpy as np

#######################################################################
# Load Files
#######################################################################

collective = pd.read_csv('./assets/title.collective4.tsv',sep='\t')
pies = pd.read_csv('./assets/title.pies.tsv',sep='\t')
full = pd.read_csv('./assets/title.full2.tsv',sep='\t')
gen = pd.read_csv('./assets/title.genreavg2.tsv',sep='\t')
year = pd.read_csv('./assets/title.yearavg.tsv',sep='\t')

#######################################################################
# Title Details and Visualizations
#######################################################################

validation_failure_message = '''
Please Try Again. The tconst provided was not found or does not exist.
'''
headers = {'User-Agent': 
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }


# request ratings distribution data for given tconst
def fetch_distribution(tconst):
    url = 'https://www.imdb.com/title/' + str(tconst) + '/ratings/'
    keys = ["props","pageProps","contentData","histogramData","histogramValues"]
    script = ''
    request = urllib.request.Request(url, headers=headers)
    voteount = []
    reason = ''

    try:
        response = urllib.request.urlopen(request)
        if response.geturl() != url:
            reason = f'Redirected to: {response.geturl()}'
        html = response.read().decode('utf-8')
        soup_ratings = BeautifulSoup(html, 'html.parser')
        script = soup_ratings.find_all('script', 
            attrs={"id": "__NEXT_DATA__", "type": "application/json"})[0]
    except urllib.error.HTTPError as e:
        reason = f'HTTPError: {e.code} - {e.reason}'
    except urllib.error.URLError as e:
        reason = f'URLError: {e.reason}'

    if script != '':
        # Extract the JSON object from the script tag
        try:
            json_data = json.loads(script.string)
            for key in keys:
                json_data = json_data.get(key, -1)
            voteCount = [json_data[i].get('voteCount', -1) for i in range(10)]
            if voteCount[0] == -1:
                reason = 'JSON key failure'
        except json.JSONDecodeError:
            reason = 'JSON decode failure'

    return voteCount if reason == '' else [reason]


# request potential production/box office details data for tconst
def fetch_details(tconst):
    url = 'https://www.imdb.com/title/' + str(tconst) + '/'
    html = ''
    request = urllib.request.Request(url, headers=headers)
    details = []
    reason = ''

    try:
        response = urllib.request.urlopen(request)
        if response.geturl() != url:
            reason = f'Redirected to: {response.geturl()}'
        html = response.read().decode('utf-8')
        soup_ratings = BeautifulSoup(html, 'html.parser')
        script = soup_ratings.find_all('script', attrs={"type": "application/ld+json"})[0]
    except urllib.error.HTTPError as e:
        reason = f'HTTPError: {e.code} - {e.reason}'
    except urllib.error.URLError as e:
        reason = f'URLError: {e.reason}'

    if html != '':
        
        json_data = json.loads(script.string)
        text = BeautifulSoup(html, 'html.parser').get_text(separator='\n')
        
        s1, f1 = text.find("Director"), text.find("See production info at IMDbPro")
        s2, f2 = text.find("Box office\nEdit") + len("Box office\nEdit"), text.find("See detailed box office info on IMDbPro")

        subtext1 = text[s1:f1] if s1 != -1 else "## Production Missing"
        subtext2 = text[s2:f2] if s2 != 14 else "\n## Box Office Missing"
        
        dividers1 = sorted([subtext1.find("Writer"), subtext1.find("Star")])[::-1]
        dividers2 = sorted([subtext2.find("Gross US"), subtext2.find("Opening"), subtext2.find("Gross worldwide")])[::-1]

        for div in dividers1:
            if div > 0:
                subtext1 = subtext1[:div] + '\n' + subtext1[div:]
        for div in dividers2:
            if div > 0:
                subtext2 = subtext2[:div] + '\n' + subtext2[div:]

        details = ["### Production:\n" + subtext1, 
                    "### Box office:" + subtext2,
                    json_data.get('contentRating', 'No Content Rating'), 
                    json_data.get('review', 'Review Not Available')]

    return details if reason == '' else ["### " + reason]


# prepare review text for display
def generate_review(review):
    if type(review) != dict:
        return ['Review Not Available','','']
    
    ratingValue = review.get('reviewRating', {'ratingValue':'?'}).get('ratingValue', '?')
    name = BeautifulSoup(review.get('name', ''), 'html.parser').get_text()
    reviewBody = review.get('reviewBody', '')
    authorName = review.get('author', {'name':'?'}).get('name', '?')
    dateCreated = review.get('dateCreated', '')
    
    heading = f"User Review - {ratingValue}/10 - {name}"
    reviewBody = BeautifulSoup(reviewBody, 'html.parser').get_text()
    footer = f"By: {authorName}, {dateCreated}"
    
    return [heading,reviewBody,footer]


# construct rating/vote distribution bar graph
def distribution_figure(name, distribution):
    bins = [str(_) for _ in range(1,11)]
    df = pd.DataFrame({ 'rating':bins, 'votes':distribution, })
    df['percent'] = ((df['votes'] / np.sum(df['votes'])) * 100
                    ).round(1).astype(str) + '%'

    fig = go.Figure(data=[go.Bar(x=df['rating'], y=df['votes'], 
        text=df['percent'], textposition='auto', width=.98, 
        marker_color='gold', marker_line=dict(width=1, color='darkgoldenrod'))])
    fig.update_layout(title=f"Voting Distribution for {name}",
            margin=dict(l=50, r=1, t=50, b=10), height=500)
    return fig


# construct scatter for titles of similar genre/year and runtime plot
def constituency_figures(title, const, genre, year):
    constituents = collective.query(
        f"(genres == '{genre}') and (year >= {year-3}) and (year <= {year+3}) and (runtime > 0)")
    size = constituents.shape[0]
    size = size if size <= 100 else 100
    constituents = constituents.sample(n=size).reset_index()

    if const not in constituents['tconst'].unique():
        constituents2 = constituents.copy()
        constituents = pd.concat([constituents, title])

    else:
        remove = constituents[constituents['index'] == title.loc[0,'index']].index[0]
        constituents2 = constituents.drop(index=remove)

    constituents['highlight'] = (constituents['tconst'] == const).replace(
        {True:title.loc[0, 'title'],False:'Constituent'})
    
    figc = px.scatter(constituents, x='votes', y='rating', 
        color='highlight', hover_data=
            {'highlight':False,'title':True,'tconst':True,'genres':True,'year':True}
        )
    figc.update_layout(margin=dict(l=5, r=5, t=25, b=10), 
        title="Constituency of Closely Related Titles", 
        showlegend=False)
    figc.update_traces(marker_size=8, marker_line=
    dict(width=2, color='rebeccapurple'), selector=dict(mode='markers'))
    
    figr = make_subplots()
    figr.add_trace(go.Box(
        x=constituents2['runtime'], 
        marker_symbol='diamond', 
        marker_color='#636efa',
        marker_size=8,
        marker_line=dict(width=2, color='rebeccapurple'),
        boxpoints='all',
        jitter=.1,
        fillcolor='rgba(255,255,255,0)',
        line_color='rgba(255,255,255,0)',
        hoveron='points',
        name='Runtimes',
        hovertext=constituents2['title']
    ), row=1, col=1)
    figr.add_trace(go.Box(
        x=title['runtime'], 
        marker_symbol='diamond', 
        marker_color='#ef553b',
        marker_size=8,
        marker_line=dict(width=2, color='rebeccapurple'),
        boxpoints='all',
        jitter=.1,
        fillcolor='rgba(255,255,255,0)',
        line_color='rgba(255,255,255,0)',
        hoveron='points',
        name='Runtimes',
        hovertext=title['title']
    ), row=1, col=1)
    figr.update_layout(showlegend=False, height=60, width=850,
        margin=go.layout.Margin(l=30, r=10, b=20, t=20, pad=4))
    figr.update_traces(marker_size=8, marker_line=
    dict(width=2, color='rebeccapurple'), selector=dict(mode='markers'))
    return figc, figr


# construct scatter for titles near in rating/votes
def neighborhood_figure(title, const, rating, votes):
    neighbors = collective.query(
        f"(rating > {rating-1.4}) and (rating <= {rating+1.4}) and (votes >= {votes-10000}) and (votes < {votes+10000})")
    size = neighbors.shape[0]
    size = size if size <= 100 else 100
    neighbors = neighbors.sample(n=size).reset_index()
    if const not in neighbors['tconst'].unique():
        neighbors = pd.concat([neighbors, title])
    neighbors['highlight'] = (neighbors['tconst'] == const).replace(
        {True:title.loc[0, 'title'],False:'Neighbor'})
    fig = px.scatter(neighbors, x='votes', y='rating', color='highlight', 
        hover_data=
        {'highlight':False,'title':True,'tconst':True,'genres':True,'year':True})
    fig.update_layout(
        margin=dict(l=5, r=5, t=25, b=10), showlegend=False,
        title="Local IMDB Neighbors")
    fig.update_traces(marker_size=8, marker_line=
    dict(width=2, color='rebeccapurple'), selector=dict(mode='markers'))
    return fig


# construct genre and year box plots and indicate title position
def comprehensives_figures(name, genre, year, rating):
    years_titles = collective.query(f"year == {year}")
    genre_titles = collective.query(f"genres == '{genre}'")
    value_to_mark = rating
    
    figy = px.box(genre_titles, y='rating')
    figy.add_shape(
        type="line", x0=-0.5, x1=0.3, y0=value_to_mark, y1=value_to_mark, 
        line=dict(
            color="Red",
            width=2,
            dash="dot", 
        ))
    figy.add_annotation(x=0.3, y=value_to_mark,
            text=f"{name[:27]}",
            showarrow=True,
            arrowhead=1
            ).update_layout(yaxis=dict(range=[1, 10]),
                margin=dict(l=5, r=5, t=25, b=5),
                title=f"{year}")
    figg = px.box(years_titles, y='rating')
    figg.add_shape(
        type="line", x0=-0.5, x1=0.3, y0=value_to_mark, y1=value_to_mark, 
        line=dict(
            color="Red",
            width=2,
            dash="dot", 
        ))
    figg.add_annotation(x=0.3, y=value_to_mark,
            text=f"{name[:27]}",
            showarrow=True,
            arrowhead=1
            ).update_layout(yaxis=dict(range=[1, 10]),
                margin=dict(l=5, r=5, t=25, b=5),
                title=f"{genre}")
    return figg, figy

# generic error indication
def error_figure(reason):
    figure = go.Figure()
    figure.add_annotation(
        text=f"Error: {str(reason)}",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=20, color="red")
        )
    figure.update_layout(title="Error in Data Processing")
    return figure

#######################################################################
# Historic Analysis Visualizations
#######################################################################

mu, sd = 6.3, 1.4

# global ratings distribution histogram
ratingfig = px.histogram(collective, 'rating', nbins=91)
ratingfig.add_shape(type='rect', x0=8.2, y0=10000, x1=9.75, y1=12000, 
    line=dict(width=1, color='black'), fillcolor='white')
ratingfig.add_shape(type="line", x0=mu, x1=mu, y0=0, y1=12000, 
    line=dict(color="Red", width=2, dash="dot"))
ratingfig.add_shape(type="line", x0=mu-sd, x1=mu+sd, y0=-30, y1=5, 
    line=dict(color="gold", width=4, ))
ratingfig.add_shape(type="line", x0=8.4, x1=8.7, y0=11400, y1=11400, 
    line=dict(color="Red", width=2, dash="dot"))
ratingfig.add_shape(type="line", x0=8.4, x1=8.7, y0=10682, y1=10718, 
    line=dict(color="gold", width=4, ))
ratingfig.add_annotation(x=8.73, y=11400, showarrow=False, text="Mean (6.3)", 
    font=dict(family="Arial", size=13, color="black"), xanchor="left")
ratingfig.add_annotation(x=8.74, y=10700, showarrow=False, text="SD (1.4)", 
    font=dict(family="Arial", size=13, color="black"), xanchor="left")
ratingfig.update_traces(marker_line_width=1,marker_line_color="rebeccapurple")
ratingfig.update_layout(margin=dict(l=50, r=1, t=50, b=10), height=500,
    title="Distribution of Global Ratings",)


# yearly average rating linegraph
voteyearfig = px.line(year, x="year", y="avg. votes")
voteyearfig.update_layout(title="Yearly Average Votes")

ratingyearfig = px.line(year, x="year", y="avg. rating")
ratingyearfig.update_layout(title="Yearly Average Rating")


# number votes cast/rating average per genre scatter plot
genrefig = px.scatter(gen, x='avg. votes', y='avg. rating', #color='genres',
    hover_data=['genres'])
genrefig.update_layout(title="Votes & Rating Averaged by Genre",
    showlegend=False)
genrefig.update_traces(marker_size=8, marker_line=
    dict(width=2, color='rebeccapurple'), selector=dict(mode='markers'))


# average runtime by genre violin plot
runavgfig = px.violin(gen,y='avg. runtime', hover_data=['genres'], 
    color_discrete_sequence=['MediumPurple'], points='all')
runavgfig.update_layout(height=1400,title="Average Runtime by Genre")


# movie information availability breakdown sunburst
sunfig = px.sunburst(
    full, 
    path=['Has Votes', 'Has Year', 'Has Runtime', 'Has Genre'], 
    values='Population', color='Fullness')
sunfig.update_layout(margin=dict(l=50, r=50, t=50, b=10),
    title="Per Page Completion Breakdown & Sunburst")


# subordinate information availability pie charts
piefig = make_subplots(rows=1, cols=4, specs=[[{'type': 'pie'}, 
    {'type': 'pie'}, {'type': 'pie'}, {'type': 'pie'}]])
# Define data for pie charts
labels1 = ['Votes', 'No Votes']
values1 = pies.iloc[0,1:]
labels2 = ['Year', 'No Year']
values2 = pies.iloc[1,1:]
labels3 = ['Runtime', 'No Runtime']
values3 = pies.iloc[2,1:]
labels4 = ['Genre', 'No Genre']
values4 = pies.iloc[3,1:]
piefig.add_trace(go.Pie(labels=labels1, values=values1, name="Pie 1"), row=1, col=1)
piefig.add_trace(go.Pie(labels=labels2, values=values2, name="Pie 2"), row=1, col=2)
piefig.add_trace(go.Pie(labels=labels3, values=values3, name="Pie 3"), row=1, col=3)
piefig.add_trace(go.Pie(labels=labels4, values=values4, name="Pie 4"), row=1, col=4)
piefig.update_layout(height=100, margin=go.layout.Margin(l=40, r=10, b=10, t=4, pad=4))

#######################################################################
# Dash App
#######################################################################

app = Dash()

app.layout = [
    html.H1(id='app-title', className='row', children='My IMDb Analytics Dashboard'
    ),
    # search box for tconst
    html.Div(id='input-div', className='row', children=[
        dcc.Input(id="input-selection", type="text", 
        placeholder="Provide a tconst", debounce=True)]
    ),
    # title view loads here
    dcc.Loading(className="loading", children=[
        html.Div(id='title', className='row')], 
        type="default"
    ),
    
    html.Hr(),
    # historic view visualizations
    html.H2(className='row', children='A Collective Examination of IMDb'
    ),

    dcc.Loading(className="loading", children=[
        html.Div(id='historic', className='row', children=[

            html.Div(className='six columns', children=[
                dcc.Graph(figure=ratingfig)
            ]),        

            html.Div(className='six columns', children=[
                dcc.Graph(figure=sunfig),
                dcc.Graph(figure=piefig)
            ])],
        )
    ]),
    # historic view visualizations lower block
    dcc.Loading(className="loading", children=[
        html.Div(id='historic2', children=[
                    
            html.Div(className='eight columns', children=[
                dcc.Graph(figure=voteyearfig),
                dcc.Graph(figure=ratingyearfig),
                dcc.Graph(figure=genrefig)
            ]),
            
            html.Div(className='four columns', children=[
                dcc.Graph(figure=runavgfig)
            ])
        
        ])
    ])
]

@callback(
    Output('title', 'children'),
    Input('input-selection', 'value'),
    prevent_initial_call=True
)
def title_view(value):
    # constructs a collection of visualizations specific to search result
    if (not value):
        return None
    if (collective.query(f"tconst == '{value}'").size == 0):
        return validation_failure_message
    
    # basic title info
    review = ['','','']
    title = collective.query(f"tconst == '{value}'").reset_index()
    titletitle = title.loc[0,'title']
    genre, year = title.loc[0, 'genres'], title.loc[0, 'year']
    rating, votes = title.loc[0, 'rating'], title.loc[0, 'votes']
    
    # request further production/box office/rating dist data
    distribution = fetch_distribution(value)
    details = fetch_details(value)
    
    if len(details) > 1:
        marktitle = f"{titletitle}"
        markmisc = f"({year}) ~ {details[2]} ~ {genre}\n\nRating: {rating}"
        markprod = details[0]
        markbox = details[1] if len(details) > 1 else "## Box Office Missing"
        review = generate_review(details[3])
    else:
        marktitle = f"{titletitle}"
        markmisc = f"({year}) ~ {genre}\n\nRating: {rating}"
        markprod = details[0]
        markbox = "## Box Office Missing"
        review = ['Review Not Available','','']

    # visualizations
    dist = distribution_figure(titletitle, distribution)   
    constituency, runs = constituency_figures(title, value, genre, year)
    neighborhood = neighborhood_figure(title, value, rating, votes)
    gencmp, yrcmp = comprehensives_figures(titletitle, genre, year, rating)
    
    # constuct new elements for view
    layout = [
        html.Div(className='row', children=[
            html.Div(className='four columns', children=[
                dcc.Link(marktitle, 
                    href=f'https://www.imdb.com/title/{value}/', target='_blank', 
                    style={'fontSize': 36}),
                dcc.Markdown(markmisc)
                ]),
            html.Div(className='four columns', children=[dcc.Markdown(markprod)]), 
            html.Div(className='four columns', children=[dcc.Markdown(markbox)])
            ], style={'color': 'black', 'fontSize': 16, 'margin': '25px'}),
        html.Div(className='row', children=[
            html.Div(className='four columns', children=[dcc.Graph(figure=constituency)]),
            html.Div(className='four columns', children=[dcc.Graph(figure=neighborhood)]),
            html.Div(className='two columns', children=[dcc.Graph(figure=gencmp)]), 
            html.Div(className='two columns', children=[dcc.Graph(figure=yrcmp)]),
            html.Div(className='eight columns', children=[
                dcc.Graph(figure=runs, config={'displayModeBar': False})
                ])
            ]),
        html.Div(className='row', children=[
            html.Div(className='six columns', children=[dcc.Graph(figure=dist)]),
            html.Div(className='six columns', children=[
                html.H3(review[0]),
                html.P(review[1], style={'maxHeight': '400px', 'overflow': 'scroll'}),
                html.H5(review[2])
                ])
            ], style={'color': 'black', 'fontSize': 16, 'textAlign': 'left'})
        ]
        
    return layout
    


if __name__ == '__main__':
    app.run(debug=True)
