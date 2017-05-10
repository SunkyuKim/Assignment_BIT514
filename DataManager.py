import pandas as pd
import logging
import ConfigParser

class DataManager(object):

    def __init__(self, loggername):
        self.logger = logging.getLogger(loggername)
        self.configParser = ConfigParser.RawConfigParser()
        self.configParser.read("config.ini")

        # Make Ready Files
        # self._load_annotation()
        # self._split_raw_table()
        # self._arrange_sample_table()
        self.__make_pure_annotation()

        annotation_pure_table = self.configParser.get(
        "preprocessed", "annotation_pure_table")
        sample_pure_table = self.configParser.get(
        "preprocessed", "sample_pure_table")
        probe_table = self.configParser.get(
        "preprocessed", "probe_table")

        self.annotation_table = pd.read_csv(annotation_pure_table)
        self.sample_table = pd.read_csv(sample_pure_table, index_col=0)
        self.probe_table = pd.read_csv(probe_table, sep ="\t")

    def __make_pure_annotation(self):
        raw_annot_file = self.configParser.get(
        "raw", "annotation_data")
        annotation_pure_table = self.configParser.get(
        "preprocessed", "annotation_pure_table")

        fr = open(raw_annot_file)
        fw = open(annotation_pure_table, "w")

        for l in fr:
            if l[0] == "#":
                continue
            fw.write(l)
        fw.close()

    def __load_annotation(self):
        """
        load probset annotation.
        :return: dict. {(probset id) : (gene symbol)}
        """
        self.logger.info("Building Annotator..")
        annotation_pure_table = self.configParser.get(
                "preprocessed", "annotation_pure_table")
        df = pd.read_csv(annotation_pure_table)

        annotator_dict = dict()
        for pid, gene in zip(list(df['Probe Set ID']), list(df['Gene Symbol'])):
            annotator_dict[pid] = gene

        self.logger.info("Build Complete")
        self.logger.debug("Size of probesets in annotator_dict : {}"
                          .format(len(annotator_dict)))
        return annotator_dict

    def __split_raw_table(self):
        """
        split data into series, sample, probe
        :return:
        """
        raw_data_file = self.configParser.get(
        "raw", "raw_data")
        series_file = self.configParser.get(
        "preprocessed", "series_file")
        sample_file = self.configParser.get(
        "preprocessed", "sample_file")
        probe_table = self.configParser.get(
        "preprocessed", "probe_table")

        self.logger.info("Raw data split -> Series, Sample, Probe")
        fr = open(raw_data_file)
        fw_series = open(series_file, 'w')
        fw_sample = open(sample_file, 'w')
        fw_probe = open(probe_table, 'w')
        for l in fr:
            tokens = l.strip().split("\t")
            if "!Series" in tokens[0]:
                fw_series.write(l)
            elif "!Sample" in tokens[0]:
                fw_sample.write(l)
            elif ("ID_REF" in tokens[0] or "_at" in tokens[0]):
                fw_probe.write(l)
        fw_series.close()
        fw_sample.close()
        fw_probe.close()
        self.logger.info("Raw data split end.")

    def __arrange_sample_table(self):
        """
        It arranges sample characteristics (!Sample_cha...) to CLEAN feature table
        :return:
        """
        self.logger.info("Making sample description to CLEAN table..")
        sample_file = self.configParser.get(
        "preprocessed", "sample_file")
        sample_pure_table = self.configParser.get(
        "preprocessed", "sample_pure_table")

        sample_dict = dict()
        for l in open(sample_file):
            tokens = l.strip().split("\t")

            if tokens[0] != "!Sample_characteristics_ch1":
                sample_dict[tokens[0]] = map(lambda x : x.replace('"', ''), tokens[1:])

            else:
                feature_value_list = map(lambda x : x.replace('"', "").split(": "), tokens[1:])
                current_featurename = ''
                for t in feature_value_list:
                    if len(t) == 2:
                        current_featurename = t[0]
                        break

                sample_dict[current_featurename] = [''] * len(feature_value_list)

                print 'current feature!', current_featurename

                for i in range(len(feature_value_list)):
                    t = feature_value_list[i]

                    if len(t) != 2:
                        continue

                    featurename, value = t

                    if featurename != current_featurename:
                        if featurename in sample_dict:
                            sample_dict[featurename][i] = value
                        else:
                            sample_dict[featurename] = [''] * len(feature_value_list)
                            sample_dict[featurename][i] = value
                        continue

                    sample_dict[featurename][i] = value

        df = pd.DataFrame(sample_dict)
        df.to_csv(sample_pure_table)
        self.logger.info("END : {}".format(sample_pure_table))

    def __make_sample_feature_table(self):
        pass

if __name__ == "__main__":
    dm = DataManager("logger")