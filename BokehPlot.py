from ConceptManager import ConceptManager as CM

from bokeh.plotting import figure, show, output_file
from bokeh.models import Label
from bokeh.palettes import brewer

class BokehPlot(object):
	"""docstring for BokehPlot"""
	colors = brewer['YlGnBu'][9]

	TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
	def __init__(self, conceptManager):
		super(BokehPlot, self).__init__()
		self.conceptManager = conceptManager
		
	def draw(self, save=False,filename="plot.png"):
		p = figure (tools=self.TOOLS)
		for i, concept in enumerate(self.conceptManager.conceptL()):
			tooltips=[('Concept',concept.conceptName())]
			x, y = concept.lowEmb()
			label = Label(x=x,y=y, text=concept.conceptName(), render_mode='css',
				border_line_color='black', border_line_alpha=1.0,
				background_fill_color='white', background_fill_alpha=1.0)
			p.scatter(x,y,color=self.colors[self.conceptManager.getCateIndex(concept.getCategory())%9],
				radius=20,fill_alpha=0.6,line_color=None)
			p.add_layout(label)

		show(p)

if __name__ == '__main__':
	cm = CM(10)
	cm.dimRed('tsne')
	BokehPlot(cm).draw()
