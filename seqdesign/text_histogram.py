#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2010 bit.ly
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""
Generate a text format histogram

This is a loose port to python of the Perl version at
http://www.pandamatak.com/people/anand/xfer/histo

http://github.com/bitly/data_hacks
https://github.com/Kobold/text_histogram
"""
import math
from dataclasses import dataclass


class MVSD(object):
    """ A class that calculates a running Mean / Variance / Standard Deviation"""
    def __init__(self):
        self.is_started = False
        self.ss = 0.0  # (running) sum of square deviations from mean
        self.m = 0.0  # (running) mean
        self.total_w = 0.0  # weight of items seen

    def add(self, x, w=1):
        """ add another datapoint to the Mean / Variance / Standard Deviation"""
        if not self.is_started:
            self.m = x
            self.ss = 0.0
            self.total_w = w
            self.is_started = True
        else:
            temp_w = self.total_w + w
            self.ss += (self.total_w * w * (x - self.m) * (x - self.m)) / temp_w
            self.m += (x - self.m) / temp_w
            self.total_w = temp_w

        # print "added %-2d mean=%0.2f var=%0.2f std=%0.2f" % (x, self.mean(), self.var(), self.sd())

    def var(self):
        return self.ss / self.total_w

    def sd(self):
        return math.sqrt(self.var())

    def mean(self):
        return self.m


def test_mvsd():
    mvsd = MVSD()
    for x in range(10):
        mvsd.add(x)

    assert '%.2f' % mvsd.mean() == "4.50"
    assert '%.2f' % mvsd.var() == "8.25"
    assert '%.14f' % mvsd.sd() == "2.87228132326901"


def median(values):
    length = len(values)
    if length % 2:
        median_indices = [length//2]
    else:
        median_indices = [length//2-1, length//2]

    values = sorted(values)
    return sum([values[i] for i in median_indices]) / len(median_indices)


def test_median():
    assert 6.0 == median([8, 7, 9, 1, 2, 6, 3])  # odd-sized list
    assert 4.5 == median([4, 5, 2, 1, 9, 10])  # even-sized int list. (4+5)/2 = 4.5
    assert "4.50" == "%.2f" % median([4.0, 5, 2, 1, 9, 10])  #even-sized float list. (4.0+5)/2 = 4.5


def histogram(stream, minimum=None, maximum=None, buckets=None, custbuckets=None, calc_mvsd=True):
    """
    Loop over the stream and add each entry to the dataset, printing out at the end


    minimum: minimum value for graph
    maximum: maximum value for graph
    buckets: Number of buckets to use for the histogram
    custbuckets: Comma seperated list of bucket edges for the histogram
    calc_mvsd: Calculate and display Mean, Variance and SD.
    """
    if not minimum or not maximum:
        # glob the iterator here so we can do min/max on it
        data = list(stream)
    else:
        data = stream
    bucket_scale = 1

    if minimum:
        min_v = minimum
    else:
        min_v = min(data)
    if maximum:
        max_v = maximum
    else:
        max_v = max(data)

    if not max_v > min_v:
        raise ValueError(f'max must be > min. max:{max_v} min:{min_v}')
    diff = max_v - min_v

    boundaries = []

    if custbuckets:
        bound = custbuckets.split(',')
        bound_sort = sorted(map(float, bound))

        # if the last value is smaller than the maximum, replace it
        if bound_sort[-1] < max_v:
            bound_sort[-1] = max_v

        # iterate through the sorted list and append to boundaries
        for x in bound_sort:
            if min_v <= x <= max_v:
                boundaries.append(x)
            elif x >= max_v:
                boundaries.append(max_v)
                break

        # beware: the min_v is not included in the boundaries, so no need to do a -1!
        bucket_counts = [0 for _ in range(len(boundaries))]
        buckets = len(boundaries)
    else:
        buckets = buckets or 10
        if buckets <= 0:
            raise ValueError('# of buckets must be > 0')
        step = diff / buckets
        bucket_counts = [0 for x in range(buckets)]
        for x in range(buckets):
            boundaries.append(min_v + (step * (x + 1)))

    skipped = 0
    samples = 0
    mvsd = MVSD()
    accepted_data = []
    for value in data:
        samples += 1
        if calc_mvsd:
            mvsd.add(value)
            accepted_data.append(value)
        # find the bucket this goes in
        if value < min_v or value > max_v:
            skipped += 1
            continue
        for bucket_postion, boundary in enumerate(boundaries):
            if value <= boundary:
                bucket_counts[bucket_postion] += 1
                break

    # auto-pick the hash scale
    if max(bucket_counts) > 75:
        bucket_scale = int(max(bucket_counts) / 75)

    return HistogramResult(
        samples=samples,
        min_v=min_v,
        max_v=max_v,
        skipped=skipped,
        calc_mvsd=calc_mvsd,
        mvsd_mean=mvsd.mean(),
        mvsd_var=mvsd.var(),
        mvsd_sd=mvsd.sd(),
        mvsd_median=median(accepted_data),
        bucket_scale=bucket_scale,
        buckets=buckets,
        boundaries=boundaries,
        bucket_counts=bucket_counts
    )


@dataclass
class HistogramResult:
    samples: int
    min_v: float
    max_v: float
    skipped: int
    calc_mvsd: bool
    mvsd_mean: float
    mvsd_var: float
    mvsd_sd: float
    mvsd_median: float
    bucket_scale: int
    buckets: int
    boundaries: list
    bucket_counts: list

    def __str__(self):
        output = [f"# NumSamples = {self.samples}; Min = {self.min_v:0.2f}; Max = {self.max_v:0.2f}"]
        if self.skipped:
            output.append(f"# {self.skipped} value{self.skipped > 1 and 's' or ''} outside of min/max")
        if self.calc_mvsd:
            output.append(f"# Mean = {self.mvsd_mean}; Variance = {self.mvsd_var}; SD = {self.mvsd_sd}; Median {self.mvsd_median}")

        output.append(f"# each ∎ represents a count of {self.bucket_scale}")
        bucket_max = self.min_v
        for bucket in range(self.buckets):
            bucket_min = bucket_max
            bucket_max = self.boundaries[bucket]
            bucket_count = self.bucket_counts[bucket]
            star_count = 0
            if bucket_count:
                star_count = bucket_count // self.bucket_scale
            output.append(f"{bucket_min:10.4f} - {bucket_max:10.4f} [{bucket_count:8d}]: {'∎' * star_count}")

        return '\n'.join(output)


class Histogram:
    """ A class that calculates a running histogram, mean, variance, and standard deviation"""
    def __init__(self, minimum, maximum, buckets=None, custbuckets=None, calc_mvsd=True):
        """
        Loop over the stream and add each entry to the dataset, printing out at the end

        minimum: minimum value for graph
        maximum: maximum value for graph
        buckets: Number of buckets to use for the histogram
        custbuckets: Comma seperated list of bucket edges for the histogram
        calc_mvsd: Calculate and display Mean, Variance and SD.
        """
        self.calc_mvsd = calc_mvsd
        self.min_v = minimum
        self.max_v = maximum

        if not self.max_v > self.min_v:
            raise ValueError(f'max must be > min. max:{self.max_v} min:{self.min_v}')
        self.diff = self.max_v - self.min_v

        boundaries = []
        if custbuckets:
            bound = custbuckets.split(',')
            bound_sort = sorted(map(float, bound))

            # if the last value is smaller than the maximum, replace it
            if bound_sort[-1] < self.max_v:
                bound_sort[-1] = self.max_v

            # iterate through the sorted list and append to boundaries
            for x in bound_sort:
                if self.min_v <= x <= self.max_v:
                    boundaries.append(x)
                elif x >= self.max_v:
                    boundaries.append(self.max_v)
                    break

            # beware: the self.min_v is not included in the boundaries, so no need to do a -1!
            bucket_counts = [0 for _ in range(len(boundaries))]
            buckets = len(boundaries)
        else:
            buckets = buckets or 10
            if buckets <= 0:
                raise ValueError('# of buckets must be > 0')
            step = self.diff / buckets
            bucket_counts = [0 for x in range(buckets)]
            for x in range(buckets):
                boundaries.append(self.min_v + (step * (x + 1)))

        self.buckets = buckets
        self.bucket_counts = bucket_counts
        self.boundaries = boundaries

        self.skipped = 0
        self.samples = 0
        self.mvsd = MVSD()

    def add(self, value):
        self.samples += 1
        if self.calc_mvsd:
            self.mvsd.add(value)

        if value < self.min_v or value > self.max_v:
            self.skipped += 1
            return
        for bucket_postion, boundary in enumerate(self.boundaries):
            if value <= boundary:
                self.bucket_counts[bucket_postion] += 1
                break

    @property
    def mvsd_mean(self):
        return self.mvsd.mean()

    @property
    def mvsd_var(self):
        return self.mvsd.var()

    @property
    def mvsd_sd(self):
        return self.mvsd.sd()

    def __str__(self):
        bucket_scale = 1
        # auto-pick the hash scale
        if max(self.bucket_counts) > 75:
            bucket_scale = int(max(self.bucket_counts) / 75)

        output = [f"# NumSamples = {self.samples}; Min = {self.min_v:0.2f}; Max = {self.max_v:0.2f}"]
        if self.skipped:
            output.append(f"# {self.skipped} value{self.skipped > 1 and 's' or ''} outside of min/max")
        if self.calc_mvsd:
            output.append(f"# Mean = {self.mvsd.mean()}; Variance = {self.mvsd.var()}; SD = {self.mvsd.sd()}")
        output.append(f"# each ∎ represents a count of {bucket_scale}")
        bucket_max = self.min_v
        for bucket in range(self.buckets):
            bucket_min = bucket_max
            bucket_max = self.boundaries[bucket]
            bucket_count = self.bucket_counts[bucket]
            star_count = 0
            if bucket_count:
                star_count = bucket_count // bucket_scale
            output.append(f"{bucket_min:10.4f} - {bucket_max:10.4f} [{bucket_count:8d}]: {'∎' * star_count}")

        return '\n'.join(output)
