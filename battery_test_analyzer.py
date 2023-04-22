import io

import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('mode.chained_assignment', None)  # Disable the warning about chained assignment


class BatteryTestAnalyzer:
    def __init__(self, file_content):
        self._file_content = file_content
        self._data_frame = self.read_data()
        self._aggregated_data_frame = self.aggregate_data()

    def read_data(self):
        return pd.read_csv(io.StringIO(self._file_content))

    def aggregate_data(self):
        """
          Aggregate the battery test data for each cycle.

          This method calculates aggregations such as Vmin, Vmax, Imax, and others
          for each cycle in the battery test data. The results are stored in a new
          DataFrame called 'aggregated_df' with a row for each cycle and columns for
          the calculated aggregations. This aggregated data can be used for
          further analysis, caching, or visualization.
          """

        aggregated_data = []

        # Add the delta time column for each measurement
        self._data_frame['dt'] = self._data_frame['test_time'].diff() / 1000  # Convert msec to sec

        # Iterate each cycle
        for cycle in self._data_frame['cycle'].unique():
            # Collect measurements which belong to the current cycle
            cycle_data = self._data_frame[self._data_frame['cycle'] == cycle]

            # Collect measurements which current is positive
            charge_data = cycle_data[cycle_data['test_cur'] > 0]

            # Collect measurements which current is negative
            discharge_data = cycle_data[cycle_data['test_cur'] < 0]

            # Collect measurements which no current is applied
            rest_data = cycle_data[cycle_data['test_cur'] == 0]

            # Minimum voltage
            v_min = cycle_data['test_vol'].min()

            # Maximum voltage
            v_max = cycle_data['test_vol'].max()

            # Maximum current
            i_max = round(cycle_data['test_cur'].abs().max() / 1000, 1)  # Convert mAmps to Amps

            # Capacity charge - Positive current over time
            cap_chg = (charge_data['test_cur'] * charge_data['dt']).sum() / 1000  # Convert mAmps to Amps

            # Capacity discharge - Negative current over time
            cap_dchg = (discharge_data['test_cur'] * discharge_data['dt']).sum() / 1000  # Convert mAmps to Amps

            # Energy charge (Power(W) = Voltage(V) * Current(A) => Energy(J) = Power(W) * Time(s))
            engy_chg = (charge_data['test_vol'] * charge_data['test_cur'] * charge_data[
                'dt']).sum() / 3600  # Convert mWh to Wh

            # Energy discharge (Power(W) = Voltage(V) * Current(A) => Energy(J) = Power(W) * Time(s))
            engy_dchg = (discharge_data['test_vol'] * discharge_data['test_cur'] * discharge_data[
                'dt']).sum() / 3600  # Convert mWh to Wh

            # Default values to cover the case that no charging or discharging is done
            ocv_drop_chg = ocv_drop_dchg = 0

            # Store the step type of the previous measurement in a new column
            cycle_data = cycle_data.copy()
            cycle_data['shifted_step_type'] = cycle_data['step_type'].shift(1)

            # Rest periods
            rest_periods = cycle_data[(cycle_data['test_cur'] == 0) & (cycle_data['step_type'] == 4)]

            """
                Grouping rest periods
            """
            # Calculate the difference between the current index and the previous index
            rest_periods['index_diff'] = rest_periods.index.to_series().diff()

            # Identify consecutive rows by comparing the calculated difference with 1
            rest_periods['is_consecutive'] = rest_periods['index_diff'] == 1

            # Assign a unique group id for each non-consecutive row using cumsum()
            # (Group id will be unique - cumsum will assign a new id for each non-consecutive row)
            rest_periods['group_id'] = (~rest_periods['is_consecutive']).cumsum()

            # Iterate rest groups
            for rest_group_id, rest_group in rest_periods.groupby('group_id'):

                # First item in group shifted_step_type is 1 or 7 (first rest after charge type 1 or 7)
                # Here 'ocv_drop_chg' will be updated if we meet a new charging phase, not sure what is the
                # order or the correct behavior of the step types.
                if rest_group.iloc[0]['shifted_step_type'] in [1, 7]:
                    ocv_drop_chg = rest_group.iloc[-1]['test_vol'] - rest_group.iloc[0]['test_vol']

                # First item in group shifted_step_type is 2 (first rest after discharge)
                if ocv_drop_dchg == 0 and rest_group.iloc[0]['shifted_step_type'] in [2]:
                    ocv_drop_dchg = rest_group.iloc[-1]['test_vol'] - rest_group.iloc[0]['test_vol']

            # Charge duration
            charge_duration = charge_data['dt'].sum()

            # CC charge time
            cc_charge_time = charge_data[charge_data['step_type'] == 1]['dt'].sum()

            # CC charge ratio
            cc_ratio = round(100 * cc_charge_time / charge_duration, 1) if charge_duration > 0 else None

            aggregated_data.append({
                'Cycle': cycle,
                'Vmin': v_min,
                'Vmax': v_max,
                'Imax': i_max,
                'Cap_Chg': cap_chg,
                'Cap_DChg': cap_dchg,
                'Engy_Chg': engy_chg,
                'Engy_DChg': engy_dchg,
                'OCV_Drop_Chg': ocv_drop_chg,
                'OCV_Drop_DChg': ocv_drop_dchg,
                'Charge_Duration': charge_duration,
                'CC_Ratio': cc_ratio
            })

        return pd.DataFrame(aggregated_data)

    def plot_aggregations(self):
        """
        Plot the aggregated battery test data in a grid of line charts.

        This method visualizes the calculated aggregations for each cycle,
        such as Vmin, Vmax, Imax, and others, as line charts with the cycle number
        on the x-axis and the aggregation values on the y-axis. The resulting
        grid of line charts provides an overview of the battery test data's
        trends and characteristics across multiple cycles.
        """

        # Define aggregation variables and their respective titles
        agg_vars = [
            ('Vmin', 'Minimal Voltage'),
            ('Vmax', 'Maximal Voltage'),
            ('Imax', 'Max Current'),
            ('Cap_Chg', 'Cap Charge'),
            ('Cap_DChg', 'Cap Discharge'),
            ('Engy_Chg', 'Energy Charge'),
            ('Engy_DChg', 'Energy Discharge'),
            ('OCV_Drop_Chg', 'OCV Drop Charge'),
            ('OCV_Drop_DChg', 'OCV Drop Discharge'),
            ('Charge_Duration', 'Charge Duration'),
            ('CC_Ratio', 'CC Ratio')
        ]

        # Define the number of rows and columns for the grid of plots
        nrows, ncols = 4, 3

        # Create a grid of subplots with the specified dimensions
        fig, axs = plt.subplots(nrows, ncols, figsize=(15, 20))

        # Loop through the aggregation variables and create a line chart for each
        for idx, (agg_var, title) in enumerate(agg_vars):
            ax = axs[idx // ncols, idx % ncols]
            ax.plot(self._aggregated_data_frame['Cycle'], self._aggregated_data_frame[agg_var], 'o-')
            ax.set_xlabel('Cycle')
            ax.set_ylabel(title)

        # Hide unused subplots if any
        for idx in range(len(agg_vars), nrows * ncols):
            axs[idx // ncols, idx % ncols].axis('off')

        # Adjust the layout and display the grid of plots
        plt.tight_layout()
        plt.show()
