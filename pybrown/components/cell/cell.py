class ColourSet:
    def __init__(self):
        """
        The colours for rendering cells
        The order of the colours here must not be changed otherwise the colours will be set
        incorrectly. If additional colours are needed they mut be appended to the end and numbers
        incremented appropriately
        """
        self.colour_map = {'PAUSE': [0.9375, 0.7383, 0.6562],
                           'GROW': [0.6562, 0.8555, 0.9375],
                           'STOPPED': [0.6680, 0.5430, 0.4883],
                           'DYING': [0.5977, 0.5859, 0.5820],
                           'STROMA': [0.9453, 0.9023, 0.6406],
                           'PILL': [0.2812, 0.6641, 0.2969],
                           'PILLGROW': [0.6641, 0.2812, 0.6484],
                           'ECOLI': [0.7578, 0.8633, 0.3359],
                           'ECOLISTOPPED': [0.4180, 0.5000, 0.0977],
                           'DIFFERENTIATED': [0.9375, 0.8945, 0.8750]}

        self.name_to_num = {key: value for value, key in enumerate(list(self.colour_map.keys()))}
        self.num_to_name = {value: key for key, value in self.name_to_num.items()}

    def get_rgb(self, c):
        """
        returns the rgb vector
        :param c:
        :return:
        """
        raise NotImplementedError

    def get_number(self, c):
        """
        Returns the number matching the name
        :param c:
        :return:
        """
        return self.name_to_num[c]



